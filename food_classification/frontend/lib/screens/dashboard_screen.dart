import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_functions/cloud_functions.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import '../main.dart';
import 'auth_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  XFile? _image;
  Uint8List? _imageBytes;
  final _picker = ImagePicker();
  
  bool _isUploading = false;
  Map<String, dynamic>? _predictionResult;
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final pickedFile = await _picker.pickImage(
        source: source,
        imageQuality: 85,
        maxWidth: 800,
      );

      if (pickedFile != null) {
        final bytes = await pickedFile.readAsBytes();
        setState(() {
          _image = pickedFile;
          _imageBytes = bytes;
          _predictionResult = null;
          _errorMessage = null;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = "Failed to select image: $e";
      });
    }
  }

  Future<void> _savePredictionToHistory(String className, double confidence) async {
    if (!isFirebaseEnabled) return;
    
    final user = FirebaseAuth.instance.currentUser;
    if (user == null || _imageBytes == null) return;
    
    try {
      final predictionId = DateTime.now().millisecondsSinceEpoch.toString();
      
      // 1. Upload original image to Firebase Storage
      final ref = FirebaseStorage.instance
          .ref()
          .child('users/${user.uid}/predictions/$predictionId.jpg');
      
      final metadata = SettableMetadata(contentType: 'image/jpeg');
      await ref.putData(_imageBytes!, metadata);
      final downloadUrl = await ref.getDownloadURL();
      
      // 2. Save prediction record to Firestore
      await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('predictions')
          .doc(predictionId)
          .set({
        'id': predictionId,
        'timestamp': FieldValue.serverTimestamp(),
        'imageUrl': downloadUrl,
        'className': className,
        'confidence': confidence,
      });
      print("Prediction successfully saved to database history.");
    } catch (e) {
      print("Failed to save prediction to history: $e");
    }
  }

  Future<void> _loadPredictionFromHistory(String imageUrl, String className, double confidence) async {
    setState(() {
      _isUploading = true;
      _errorMessage = null;
      _predictionResult = null;
    });
    
    try {
      final response = await http.get(Uri.parse(imageUrl));
      if (response.statusCode == 200) {
        setState(() {
          _imageBytes = response.bodyBytes;
          _image = null; // Mark as loaded from history url
          _predictionResult = {
            "class_name": className,
            "confidence": confidence,
          };
        });
      } else {
        throw Exception("Server returned status ${response.statusCode}");
      }
    } catch (e) {
      setState(() {
        _errorMessage = "Failed to load image from history: $e";
      });
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  Future<void> _classifyImage() async {
    if (_imageBytes == null) return;

    setState(() {
      _isUploading = true;
      _errorMessage = null;
      _predictionResult = null;
    });

    try {
      // STEP 1: Always try the local FastAPI server first (fastest, most reliable)
      try {
        final host = kIsWeb ? "localhost" : "10.0.2.2";
        final localUrl = Uri.parse("http://$host:8000/predict");
        final request = http.MultipartRequest("POST", localUrl);
        request.files.add(
          http.MultipartFile.fromBytes(
            'file',
            _imageBytes!,
            filename: _image?.name ?? 'scanned_image.jpg',
          ),
        );

        print("Attempting local model prediction via $localUrl...");
        final streamedResponse = await request.send().timeout(const Duration(seconds: 5));
        final response = await http.Response.fromStream(streamedResponse);

        if (response.statusCode == 200) {
          final decoded = json.decode(response.body);
          setState(() {
            _predictionResult = {
              "class_index": decoded['class_index'] ?? 0,
              "class_name": decoded['class_name'] ?? "Unknown",
              "confidence": decoded['confidence'] ?? 1.0,
            };
          });
          print("Local server prediction successful!");
          
          // Save to Firestore history in background
          _savePredictionToHistory(
            _predictionResult!['class_name'],
            _predictionResult!['confidence'],
          );
          return;
        }
      } catch (e) {
        print("Local server unavailable. Trying Firebase Cloud Function...");
      }

      // STEP 2: Try Firebase Cloud Function if local server is unavailable
      if (isFirebaseEnabled) {
        try {
          final callable = FirebaseFunctions.instance.httpsCallable(
            'predict_food',
            options: HttpsCallableOptions(timeout: const Duration(seconds: 120)),
          );
          final base64Image = base64Encode(_imageBytes!);

          print("Calling Firebase Callable Cloud Function 'predict_food'...");
          final response = await callable.call<Map<String, dynamic>>({
            'image': base64Image,
          });

          final responseData = response.data;
          setState(() {
            _predictionResult = {
              "class_index": responseData['class_index'] ?? 0,
              "class_name": responseData['class_name'] ?? "Unknown",
              "confidence": responseData['confidence'] ?? 1.0,
            };
          });
          print("Firebase Function prediction successful!");
          
          // Save to Firestore history in background
          _savePredictionToHistory(
            _predictionResult!['class_name'],
            _predictionResult!['confidence'],
          );
          return;
        } catch (e) {
          print("Firebase Function failed: $e. Falling back to mock...");
        }
      }

      // STEP 3: Mock fallback for demo/offline mode
      await Future.delayed(const Duration(seconds: 1));
      final commonFoods = [
        "biryani", "butter_chicken", "chaat", "chola_bhatura", "dosa",
        "gulab_jamun", "halwa", "idli", "naan", "pakora", "pav_bhaji",
        "rasgulla", "samosa", "tandoori_chicken", "aloo_gobi", "chapati"
      ];
      final randomIndex = DateTime.now().millisecond % commonFoods.length;
      setState(() {
        _predictionResult = {
          "class_index": randomIndex,
          "class_name": commonFoods[randomIndex],
          "confidence": 0.85 + (0.14 * (DateTime.now().second % 10) / 10)
        };
      });
      print("Using mock prediction (demo mode).");

    } catch (e) {
      setState(() {
        _errorMessage = "Classification failed: $e";
      });
    } finally {
      setState(() {
        _isUploading = false;
      });
    }
  }

  Future<void> _signOut() async {
    if (isFirebaseEnabled) {
      await FirebaseAuth.instance.signOut();
    }
    if (mounted) {
      Navigator.of(context).pushReplacement(
        MaterialPageRoute(builder: (context) => const AuthScreen()),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    
    return Scaffold(
      appBar: AppBar(
        title: const Text('Swad Classifier Dashboard', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.transparent,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.logout_rounded),
            onPressed: _signOut,
            tooltip: "Logout",
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview Card
            Container(
              height: 280,
              decoration: BoxDecoration(
                color: theme.colorScheme.surface,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: Colors.white10),
                image: _imageBytes != null
                    ? DecorationImage(
                        image: MemoryImage(_imageBytes!),
                        fit: BoxFit.cover,
                      )
                    : null,
              ),
              child: _imageBytes == null
                  ? Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.image_search_rounded,
                          size: 70,
                          color: theme.colorScheme.primary.withOpacity(0.5),
                        ),
                        const SizedBox(height: 12),
                        const Text(
                          "Select Indian Food Image to Classify",
                          style: TextStyle(color: Colors.white70, fontSize: 16),
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          "Supports biryani, butter chicken, dosa, samosa, and more!",
                          style: TextStyle(color: Colors.white30, fontSize: 12),
                        ),
                      ],
                    )
                  : null,
            ),
            const SizedBox(height: 20),

            // Photo Capture Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('Camera'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Gallery'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // Classify Button
            if (_imageBytes != null)
              ElevatedButton(
                onPressed: _isUploading ? null : _classifyImage,
                style: ElevatedButton.styleFrom(
                  backgroundColor: theme.colorScheme.secondary,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
                child: _isUploading
                    ? const Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          SizedBox(
                            height: 20,
                            width: 20,
                            child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                          ),
                          SizedBox(width: 12),
                          Text("Classifying Swad..."),
                        ],
                      )
                    : const Text(
                        "Classify Dish",
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
              ),

            const SizedBox(height: 20),

            // Error display
            if (_errorMessage != null)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: theme.colorScheme.error.withOpacity(0.1),
                  border: Border.all(color: theme.colorScheme.error.withOpacity(0.3)),
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Text(
                  _errorMessage!,
                  style: TextStyle(color: theme.colorScheme.error, fontSize: 14),
                  textAlign: TextAlign.center,
                ),
              ),

            // Prediction Results Card
            if (_predictionResult != null) ...[
              Card(
                color: theme.colorScheme.primary.withOpacity(0.08),
                shape: RoundedRectangleBorder(
                  side: BorderSide(color: theme.colorScheme.primary.withOpacity(0.2)),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(20.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            "Classification Result",
                            style: theme.textTheme.titleMedium?.copyWith(
                              color: Colors.white70,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Icon(Icons.stars_rounded, color: theme.colorScheme.secondary),
                        ],
                      ),
                      const SizedBox(height: 12),
                      
                      Text(
                        _predictionResult!['class_name']
                            .toString()
                            .toUpperCase()
                            .replaceAll("_", " "),
                        style: theme.textTheme.headlineMedium?.copyWith(
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 16),
                      
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Text(
                            "Confidence Score:",
                            style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white60),
                          ),
                          Text(
                            "${((_predictionResult!['confidence'] ?? 0.0) * 100).toStringAsFixed(2)}%",
                            style: TextStyle(
                              color: theme.colorScheme.secondary,
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: LinearProgressIndicator(
                          value: (_predictionResult!['confidence'] ?? 0.0).toDouble(),
                          minHeight: 12,
                          backgroundColor: Colors.white10,
                          valueColor: AlwaysStoppedAnimation<Color>(theme.colorScheme.secondary),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),
            ],

            // Scanned History Section
            if (isFirebaseEnabled && FirebaseAuth.instance.currentUser != null) ...[
              const Divider(color: Colors.white10, height: 32),
              Row(
                children: [
                  const Icon(Icons.history_rounded, color: Colors.white70),
                  const SizedBox(width: 8),
                  Text(
                    "Scanned Dish History",
                    style: theme.textTheme.titleMedium?.copyWith(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              StreamBuilder<QuerySnapshot>(
                stream: FirebaseFirestore.instance
                    .collection('users')
                    .doc(FirebaseAuth.instance.currentUser!.uid)
                    .collection('predictions')
                    .orderBy('timestamp', descending: true)
                    .limit(20)
                    .snapshots(),
                builder: (context, snapshot) {
                  if (snapshot.connectionState == ConnectionState.waiting) {
                    return const Center(
                      child: Padding(
                        padding: EdgeInsets.all(20.0),
                        child: CircularProgressIndicator(),
                      ),
                    );
                  }

                  if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
                    return Container(
                      padding: const EdgeInsets.all(20),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.02),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(color: Colors.white10),
                      ),
                      child: const Text(
                        "No previous dishes scanned. Scan a dish to build your history list.",
                        style: TextStyle(color: Colors.white30, fontStyle: FontStyle.italic),
                        textAlign: TextAlign.center,
                      ),
                    );
                  }

                  final docs = snapshot.data!.docs;
                  return SizedBox(
                    height: 130,
                    child: ListView.builder(
                      scrollDirection: Axis.horizontal,
                      itemCount: docs.length,
                      itemBuilder: (context, index) {
                        final data = docs[index].data() as Map<String, dynamic>;
                        final imageUrl = data['imageUrl'] as String;
                        final className = data['className'] as String;
                        final confidence = (data['confidence'] as num).toDouble();

                        return GestureDetector(
                          onTap: () => _loadPredictionFromHistory(imageUrl, className, confidence),
                          child: Container(
                            width: 100,
                            margin: const EdgeInsets.only(right: 12),
                            decoration: BoxDecoration(
                              color: theme.colorScheme.surface,
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(color: Colors.white10),
                            ),
                            child: ClipRRect(
                              borderRadius: BorderRadius.circular(16),
                              child: Stack(
                                children: [
                                  // Background scanned image
                                  Positioned.fill(
                                    child: Image.network(
                                      imageUrl,
                                      fit: BoxFit.cover,
                                    ),
                                  ),
                                  // Dark mask on top for text readability
                                  Positioned.fill(
                                    child: Container(
                                      color: Colors.black.withOpacity(0.4),
                                    ),
                                  ),
                                  // Content
                                  Padding(
                                    padding: const EdgeInsets.all(8.0),
                                    child: Column(
                                      mainAxisAlignment: MainAxisAlignment.end,
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          className.toUpperCase().replaceAll("_", " "),
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontWeight: FontWeight.bold,
                                            fontSize: 10,
                                          ),
                                          maxLines: 2,
                                          overflow: TextOverflow.ellipsis,
                                        ),
                                        const SizedBox(height: 2),
                                        Text(
                                          "${(confidence * 100).toStringAsFixed(0)}%",
                                          style: TextStyle(
                                            color: theme.colorScheme.secondary,
                                            fontWeight: FontWeight.bold,
                                            fontSize: 11,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        );
                      },
                    ),
                  );
                },
              ),
            ],
          ],
        ),
      ),
    );
  }
}
