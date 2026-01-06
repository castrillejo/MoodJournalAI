import { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';
import { motion } from 'framer-motion';
import TextInput from './components/TextInput';
import ResultCard from './components/ResultCard';
import { predictEmotion, predictEmotionWithAttention, checkAPIHealth } from './services/api';

function App() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showAttention, setShowAttention] = useState(false);
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    const checkHealth = async () => {
      const health = await checkAPIHealth();
      setApiStatus(health.status === 'healthy' ? 'online' : 'offline');
    };

    checkHealth();
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
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
      {/* Header */}
      <header className="bg-slate-950/70 backdrop-blur-sm sticky top-0 z-50 border-b border-slate-800">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3"
            >
              <div className="text-4xl">🎭</div>
              <div>
                <h1 className="text-2xl font-bold text-slate-100">
                  MoodJournalAI
                </h1>
                <p className="text-sm text-slate-300">
                  Emotion Classification with AI
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900/70 border border-slate-800"
            >
              <Activity
                className={`w-4 h-4 ${apiStatus === 'online'
                  ? 'text-green-400 animate-pulse'
                  : apiStatus === 'checking'
                    ? 'text-yellow-300'
                    : 'text-red-400'
                  }`}
              />
              <span className="text-sm font-medium text-slate-200">
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
          <h2 className="text-5xl font-bold text-slate-100 mb-4">
            Discover the Emotions in Your Text
          </h2>
          <p className="text-xl text-slate-300 max-w-2xl mx-auto">
            Powered by <span className="font-semibold text-purple-300">RoBERTa</span> fine-tuned for emotion detection.
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
            className="max-w-4xl mx-auto mt-8 p-6 bg-red-950/40 border border-red-900 rounded-2xl"
          >
            <p className="text-red-200 text-center font-medium">{error}</p>
            <p className="text-red-300 text-center text-sm mt-2">
              Run: <code className="bg-red-950/70 border border-red-900 px-2 py-1 rounded">uvicorn backend.api.app.main:app --reload</code>
            </p>
          </motion.div>
        )}

        {/* Results */}
        {result && <ResultCard result={result} showAttention={showAttention} />}
      </main>

      {/* Footer */}
      <footer className="bg-slate-950/50 backdrop-blur-sm mt-20 py-8 border-t border-slate-800">
        <div className="container mx-auto px-4 text-center text-slate-400">
          <p className="text-sm">
            Developed with ❤️ using <span className="font-semibold text-slate-200">RoBERTa Fine-tuning</span> | MoodJournalAI v2.0
          </p>
          <p className="text-xs mt-2 text-slate-500">
            Technologies: React · FastAPI · Recharts · Framer Motion · TailwindCSS
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
