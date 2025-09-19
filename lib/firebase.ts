import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
import { getFunctions } from "firebase/functions";

// Your web app's Firebase configuration
// IMPORTANT: In a production application, these keys should be stored securely,
// for example, in environment variables.
const firebaseConfig = {
  apiKey: "AIzaSyAgJM9aIaT9Ye33XHu39vMMnuAqb2o9xC8",
  authDomain: "ship-routing-app.firebaseapp.com",
  projectId: "ship-routing-app",
  storageBucket: "ship-routing-app.appspot.com",
  messagingSenderId: "179380437829",
  appId: "1:179380437829:web:a9f4c24ead9c717a50742c"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const storage = getStorage(app);
const functions = getFunctions(app);

// If you are running the Firebase emulator for local development, uncomment this line.
// connectFunctionsEmulator(functions, "localhost", 5001);

export { app, db, storage, functions };
