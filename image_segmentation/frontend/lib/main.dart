import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/auth_screen.dart';

// Global flag to track if Firebase is successfully configured
bool isFirebaseEnabled = false;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  try {
    await Firebase.initializeApp(
      options: const FirebaseOptions(
        apiKey: "AIzaSyDJk0BFivFQggKuBcoMHYvxuVWcfuTYa5Q",
        authDomain: "image-segmentation-2b428.firebaseapp.com",
        projectId: "image-segmentation-2b428",
        storageBucket: "image-segmentation-2b428.firebasestorage.app",
        messagingSenderId: "265954523093",
        appId: "1:265954523093:web:27d2712348c262563a6f5c",
        measurementId: "G-73Y65ZNK1J",
      ),
    );
    isFirebaseEnabled = true;
    print("Firebase successfully initialized.");
  } catch (e) {
    print("WARNING: Firebase initialization failed: $e");
    print("Running in Local Demo Mode (Firebase Authentication bypassed).");
  }

  runApp(const FoodClassifierApp());
}

class FoodClassifierApp extends StatelessWidget {
  const FoodClassifierApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Swad Segmenter',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF9C27B0), // Premium Purple
          primary: const Color(0xFF9C27B0),
          secondary: const Color(0xFF00E5FF), // Cyan accent
          brightness: Brightness.dark,
          background: const Color(0xFF121212),
          surface: const Color(0xFF1E1E1E),
        ),
        textTheme: GoogleFonts.outfitTextTheme(
          ThemeData.dark().textTheme,
        ),
        cardTheme: CardThemeData(
          color: const Color(0xFF1E1E1E),
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
          ),
        ),
      ),
      home: const AuthScreen(),
    );
  }
}
