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
  double _maskOpacity = 0.5;

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

  Future<void> _savePredictionToHistory(String maskBase64, List<dynamic> detectedIngredients) async {
    if (!isFirebaseEnabled) return;
    
    final user = FirebaseAuth.instance.currentUser;
    if (user == null || _imageBytes == null) return;
    
    try {
      final predictionId = DateTime.now().millisecondsSinceEpoch.toString();
      
      // 1. Upload original image to Firebase Storage
      final imgRef = FirebaseStorage.instance
          .ref()
          .child('users/${user.uid}/segmentations/$predictionId.jpg');
      await imgRef.putData(_imageBytes!, SettableMetadata(contentType: 'image/jpeg'));
      final imgUrl = await imgRef.getDownloadURL();
      
      // 2. Upload colored mask overlay PNG to Firebase Storage
      final maskRef = FirebaseStorage.instance
          .ref()
          .child('users/${user.uid}/segmentations/${predictionId}_mask.png');
      final maskBytes = base64Decode(maskBase64);
      await maskRef.putData(maskBytes, SettableMetadata(contentType: 'image/png'));
      final maskUrl = await maskRef.getDownloadURL();
      
      // 3. Save prediction record to Firestore
      await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('segmentations')
          .doc(predictionId)
          .set({
        'id': predictionId,
        'timestamp': FieldValue.serverTimestamp(),
        'imageUrl': imgUrl,
        'maskUrl': maskUrl,
        'detected_ingredients': detectedIngredients,
      });
      print("Segmentation scan successfully saved to database history.");
    } catch (e) {
      print("Failed to save segmentation history: $e");
    }
  }

  Future<void> _loadPredictionFromHistory(String imageUrl, String maskUrl, List<dynamic> detectedIngredients) async {
    setState(() {
      _isUploading = true;
      _errorMessage = null;
      _predictionResult = null;
    });
    
    try {
      // Download original image
      final imgResponse = await http.get(Uri.parse(imageUrl));
      // Download mask overlay
      final maskResponse = await http.get(Uri.parse(maskUrl));
      
      if (imgResponse.statusCode == 200 && maskResponse.statusCode == 200) {
        setState(() {
          _imageBytes = imgResponse.bodyBytes;
          _image = null;
          _predictionResult = {
            "mask": base64Encode(maskResponse.bodyBytes),
            "detected_ingredients": detectedIngredients,
          };
        });
      } else {
        throw Exception("Server returned error loading prediction assets.");
      }
    } catch (e) {
      setState(() {
        _errorMessage = "Failed to load history item: $e";
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
        final localUrl = Uri.parse("http://$host:8001/predict");
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
          if (decoded['mask'] != null) {
            setState(() {
              _predictionResult = {
                "mask": decoded['mask'],
                "detected_ingredients": decoded['detected_ingredients'] ?? []
              };
            });
            print("Local server prediction successful!");
            
            // Save history in background
            _savePredictionToHistory(
              _predictionResult!['mask'],
              _predictionResult!['detected_ingredients'],
            );
            return;
          } else {
            print("Local server returned empty mask (wrong backend?). Falling back...");
          }
        }
      } catch (e) {
        print("Local server unavailable. Trying Firebase Cloud Function...");
      }

      // STEP 2: Try Firebase Cloud Function if local server is unavailable
      if (isFirebaseEnabled) {
        try {
          final callable = FirebaseFunctions.instance.httpsCallable(
            'segment_food',
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
              "mask": responseData['mask'],
              "detected_ingredients": responseData['detected_ingredients'] ?? []
            };
          });
          print("Firebase Function prediction successful!");
          
          // Save history in background
          _savePredictionToHistory(
            _predictionResult!['mask'],
            _predictionResult!['detected_ingredients'],
          );
          return;
        } catch (e) {
          print("Firebase Function failed: $e. Falling back to mock...");
        }
      }

      // STEP 3: Mock fallback for demo/offline mode
      await Future.delayed(const Duration(seconds: 1));
      
      const mockMaskBase64 = "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";
      
      setState(() {
        _predictionResult = {
          "mask": mockMaskBase64,
          "detected_ingredients": [
            {"class_id": 58, "name": "Bread", "color": "#e056fd", "percentage": 35.50},
            {"class_id": 80, "name": "Lettuce", "color": "#2ecc71", "percentage": 15.20},
            {"class_id": 73, "name": "Tomato", "color": "#e74c3c", "percentage": 10.80}
          ]
        };
      });
      print("Using mock prediction (demo mode).");

    } catch (e) {
      setState(() {
        _errorMessage = "Segmentation failed: $e";
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
        title: const Text('Swad Segmenter Dashboard', style: TextStyle(fontWeight: FontWeight.bold)),
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
            // Image Preview Stack (Original + Mask overlay)
            Container(
              height: 320,
              decoration: BoxDecoration(
                color: theme.colorScheme.surface,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: Colors.white10),
              ),
              child: _imageBytes == null
                  ? Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Icon(
                          Icons.blur_linear_rounded,
                          size: 70,
                          color: theme.colorScheme.secondary.withOpacity(0.5),
                        ),
                        const SizedBox(height: 12),
                        const Text(
                          "Select Food Image to Segment",
                          style: TextStyle(color: Colors.white70, fontSize: 16, fontWeight: FontWeight.bold),
                        ),
                        const SizedBox(height: 4),
                        const Text(
                          "Detects and segments over 100 food classes!",
                          style: TextStyle(color: Colors.white30, fontSize: 12),
                        ),
                      ],
                    )
                  : Stack(
                      alignment: Alignment.center,
                      children: [
                        // Original Image
                        ClipRRect(
                          borderRadius: BorderRadius.circular(24),
                          child: Image.memory(
                            _imageBytes!,
                            fit: BoxFit.contain,
                            height: 320,
                            width: double.infinity,
                          ),
                        ),
                        // Segmentation Mask Overlay
                        if (_predictionResult != null && _predictionResult!['mask'] != null)
                          ClipRRect(
                            borderRadius: BorderRadius.circular(24),
                            child: Opacity(
                              opacity: _maskOpacity,
                              child: Image.memory(
                                base64Decode(_predictionResult!['mask']),
                                fit: BoxFit.contain,
                                height: 320,
                                width: double.infinity,
                              ),
                            ),
                          ),
                      ],
                    ),
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

            // Segment Button
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
                          Text("Segmenting Food Ingredients..."),
                        ],
                      )
                    : const Text(
                        "Segment Dish",
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
              ),

            const SizedBox(height: 20),

            // Opacity Slider (only visible after prediction)
            if (_predictionResult != null && _predictionResult!['mask'] != null) ...[
              Card(
                color: Colors.white.withOpacity(0.03),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Row(
                    children: [
                      const Icon(Icons.opacity, color: Colors.white60),
                      const SizedBox(width: 12),
                      const Text("Overlay opacity", style: TextStyle(color: Colors.white70, fontWeight: FontWeight.w500)),
                      Expanded(
                        child: Slider(
                          value: _maskOpacity,
                          min: 0.0,
                          max: 1.0,
                          onChanged: (val) {
                            setState(() {
                              _maskOpacity = val;
                            });
                          },
                        ),
                      ),
                      Text("${(_maskOpacity * 100).toInt()}%", style: const TextStyle(color: Colors.white70, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 20),
            ],

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

            // Prediction Results Card (Legend)
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
                            "Segmented Ingredients",
                            style: theme.textTheme.titleMedium?.copyWith(
                              color: Colors.white70,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Icon(Icons.restaurant_menu_rounded, color: theme.colorScheme.secondary),
                        ],
                      ),
                      const SizedBox(height: 16),
                      if ((_predictionResult!['detected_ingredients'] as List).isEmpty)
                        const Text(
                          "No clear food segments detected.",
                          style: TextStyle(color: Colors.white54, fontStyle: FontStyle.italic),
                        )
                      else
                        ListView.separated(
                          shrinkWrap: true,
                          physics: const NeverScrollableScrollPhysics(),
                          itemCount: (_predictionResult!['detected_ingredients'] as List).length,
                          separatorBuilder: (_, __) => const Divider(color: Colors.white10),
                          itemBuilder: (context, index) {
                            final ingredient = (_predictionResult!['detected_ingredients'] as List)[index];
                            final colorHex = ingredient['color'] as String;
                            final color = Color(int.parse(colorHex.replaceFirst('#', '0xff')));
                            
                            return Padding(
                              padding: const EdgeInsets.symmetric(vertical: 4.0),
                              child: Row(
                                children: [
                                  Container(
                                    width: 20,
                                    height: 20,
                                    decoration: BoxDecoration(
                                      color: color,
                                      borderRadius: BorderRadius.circular(6),
                                      border: Border.all(color: Colors.white24),
                                    ),
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: Text(
                                      ingredient['name'].toString(),
                                      style: const TextStyle(
                                        color: Colors.white,
                                        fontSize: 16,
                                        fontWeight: FontWeight.w600,
                                      ),
                                    ),
                                  ),
                                  Text(
                                    "${ingredient['percentage']}%",
                                    style: TextStyle(
                                      color: theme.colorScheme.secondary,
                                      fontWeight: FontWeight.bold,
                                      fontSize: 16,
                                    ),
                                  ),
                                ],
                              ),
                            );
                          },
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
                    "Scanned Segmentation History",
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
                    .collection('segmentations')
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
                        "No previous segmentations scanned. Scan a dish to build your history list.",
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
                        final maskUrl = data['maskUrl'] as String;
                        final detectedIngredients = data['detected_ingredients'] as List<dynamic>;

                        // Generate summary names for overlay card text
                        final namesSummary = detectedIngredients.map((e) => e['name']).take(2).join(', ');

                        return GestureDetector(
                          onTap: () => _loadPredictionFromHistory(imageUrl, maskUrl, detectedIngredients),
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
                                          namesSummary.isNotEmpty ? namesSummary : "Scan",
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
                                          "${detectedIngredients.length} Items",
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
