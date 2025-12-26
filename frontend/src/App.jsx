import { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';
import { motion } from 'framer-motion';
import TextInput from './components/TextInput';
import ResultCard from './components/ResultCard';
import { predictEmotion, predictEmotionWithAttention, checkAPIHealth } from './services/api';
import './App.css';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAttention, setShowAttention] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    // Check API health on mount
    const checkHealth = async () => {
      const health = await checkAPIHealth();
      setApiStatus(health.status === 'healthy' ? 'online' : 'offline');
    };
    checkHealth();

    // Check every 30 seconds
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const handleAnalyze = async (text) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = showAttention
        ? await predictEmotionWithAttention(text)
        : await predictEmotion(text);
      setResult(data);
    } catch (err) {
      setError('Failed to analyze emotion. Make sure the API server is running on http://localhost:8000');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm shadow-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3"
            >
              <div className="text-4xl">🎭</div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">
                  MoodJournalAI
                </h1>
                <p className="text-sm text-gray-600">
                  Emotion Classification with AI
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-gray-100"
            >
              <Activity className={`w-4 h-4 ${apiStatus === 'online' ? 'text-green-500 animate-pulse' : 'text-red-500'}`} />
              <span className="text-sm font-medium">
                API: {apiStatus === 'online' ? 'Online' : apiStatus === 'checking' ? 'Checking...' : 'Offline'}
              </span>
            </motion.div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h2 className="text-5xl font-bold text-gray-800 mb-4">
            Discover the Emotions in Your Text
          </h2>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Powered by <span className="font-semibold text-purple-600">RoBERTa</span> fine-tuned for emotion detection.
            Supports 6 emotions with optional attention visualization.
          </p>
        </motion.div>

        {/* Input Section */}
        <TextInput
          onAnalyze={handleAnalyze}
          isLoading={isLoading}
          showAttention={showAttention}
          setShowAttention={setShowAttention}
        />

        {/* Error Message */}
        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-4xl mx-auto mt-8 p-6 bg-red-50 border-2 border-red-200 rounded-2xl"
          >
            <p className="text-red-800 text-center font-medium">{error}</p>
            <p className="text-red-600 text-center text-sm mt-2">
              Run: <code className="bg-red-100 px-2 py-1 rounded">uvicorn backend.api.app.main:app --reload</code>
            </p>
          </motion.div>
        )}

        {/* Results */}
        {result && <ResultCard result={result} showAttention={showAttention} />}
      </main>

      {/* Footer */}
      <footer className="bg-white/50 backdrop-blur-sm mt-20 py-8">
        <div className="container mx-auto px-4 text-center text-gray-600">
          <p className="text-sm">
            Developed with ❤️ using <span className="font-semibold">RoBERTa Fine-tuning</span> | MoodJournalAI v2.0
          </p>
          <p className="text-xs mt-2">
            Technologies: React · FastAPI · Recharts · Framer Motion · TailwindCSS
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
