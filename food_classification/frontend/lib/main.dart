import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/auth_screen.dart';

// Global flag to track if Firebase is successfully configured
bool isFirebaseEnabled = false;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  try {
    // Initialize Firebase explicitly using Web configurations
    await Firebase.initializeApp(
      options: const FirebaseOptions(
        apiKey: "AIzaSyCoXgdMa0Di5nF4t4lGnK90UxycPzhkeiI",
        authDomain: "image-classification-be40c.firebaseapp.com",
        projectId: "image-classification-be40c",
        storageBucket: "image-classification-be40c.firebasestorage.app",
        messagingSenderId: "841150950761",
        appId: "1:841150950761:web:f7c306800fb988b457e1f8",
        measurementId: "G-X8RYL02KMK",
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
      title: 'Indian Food Classifier',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFFF5722), // Vibrant Orange
          primary: const Color(0xFFFF5722),
          secondary: const Color(0xFF4CAF50), // Healthy Green
          brightness: Brightness.dark, // Sleek Dark Mode
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
