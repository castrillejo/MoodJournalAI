import { useEffect, useMemo, useState } from 'react';
import { Activity } from 'lucide-react';
import { motion } from 'framer-motion';

import { checkAPIHealth, fetchEvaluationOverview, predictCompareWithAttention } from './services/api';
import ModelOverviewSection from './components/ModelOverviewSection';
import ComparePredictionsSection from './components/ComparePredictionsSection';
import UsersSection from "./components/UsersSection";

function App() {
  const [apiStatus, setApiStatus] = useState('checking');

  const [overview, setOverview] = useState(null);
  const [overviewLoading, setOverviewLoading] = useState(false);
  const [overviewError, setOverviewError] = useState(null);

  const [selectedSemi, setSelectedSemi] = useState('semi_frozen4');

  const [compareResult, setCompareResult] = useState(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState(null);

  // Health check
  useEffect(() => {
    const checkHealth = async () => {
      const health = await checkAPIHealth();
      setApiStatus(health.status === 'healthy' ? 'online' : 'offline');
    };
    checkHealth();

    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Load overview once
  useEffect(() => {
    const load = async () => {
      setOverviewLoading(true);
      setOverviewError(null);

      try {
        const data = await fetchEvaluationOverview();
        setOverview(data);

        // Default semi selection if available
        const variants = data?.models?.semi_frozen?.variants || {};
        const keys = Object.keys(variants);
        if (keys.length) {
          // prefer 4 if exists, else first key
          setSelectedSemi((prev) => {
            if (prev && keys.includes(prev)) return prev;
            if (keys.includes('semi_frozen4')) return 'semi_frozen4';
            return keys[0];
          });
        }
      } catch (err) {
        setOverviewError(
          'Failed to load evaluation overview. Make sure the API server is running on http://localhost:8000'
        );
        console.error(err);
      } finally {
        setOverviewLoading(false);
      }
    };

    load();
  }, []);

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const handleAnalyzeCompare = async (text) => {
    setCompareLoading(true);
    setCompareError(null);
    setCompareResult(null);

    try {
      const data = await predictCompareWithAttention(text, selectedSemi);
      // esperamos { frozen, semi, finetuned } (o algo equivalente)
      setCompareResult(data);
    } catch (err) {
      setCompareError(
        'Failed to run comparison. If you do not have the compare endpoint yet, ensure /predict/attention supports the "model" field.'
      );
      console.error(err);
    } finally {
      setCompareLoading(false);
    }
  };

  const headerStatusIconClass = useMemo(() => {
    if (apiStatus === 'online') return 'text-green-500 animate-pulse';
    if (apiStatus === 'checking') return 'text-gray-400';
    return 'text-red-500';
  }, [apiStatus]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950">
      {/* Header */}
      <header className="bg-slate-950/70 backdrop-blur-sm border-b border-slate-800 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-3"
            >
              <div className="text-4xl">🎭</div>
              <div>
                <h1 className="text-2xl font-bold text-gray-800">MoodJournalAI</h1>
                <p className="text-sm text-gray-600">Clasifica hasta 6 emociones</p>
              </div>
            </motion.div>

            <div className="flex items-center gap-3 flex-wrap">
              {/* API Status */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                className="flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900/70 border border-slate-800"
              >
                <Activity className={`w-4 h-4 ${headerStatusIconClass}`} />
                <span className="text-sm font-medium">
                  API: {apiStatus === 'online' ? 'Online' : apiStatus === 'checking' ? 'Checking...' : 'Offline'}
                </span>
              </motion.div>
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="container mx-auto px-4 py-10">
        {/* Overview loading/error */}
        {overviewLoading && (
          <div className="max-w-6xl mx-auto mb-6 p-6 bg-white rounded-2xl shadow-lg border border-gray-100 shimmer">
            <p className="text-gray-600 font-medium">Loading evaluation overview...</p>
          </div>
        )}

        {overviewError && (
          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="max-w-6xl mx-auto mb-6 p-6 bg-red-50 border-2 border-red-200 rounded-2xl"
          >
            <p className="text-red-800 text-center font-medium">{overviewError}</p>
            <p className="text-red-600 text-center text-sm mt-2">
              Run: <code className="bg-red-100 px-2 py-1 rounded">uvicorn backend.api.app.main:app --reload</code>
            </p>
          </motion.div>
        )}

        {/* Section 1: Overview */}
        {overview && (
          <ModelOverviewSection
            overview={overview}
            selectedSemi={selectedSemi}
            setSelectedSemi={setSelectedSemi}
          />
        )}

        {/* Section 2: Compare */}
        <ComparePredictionsSection
          selectedSemi={selectedSemi}
          onAnalyzeCompare={handleAnalyzeCompare}
          compareLoading={compareLoading}
          compareError={compareError}
          compareResult={compareResult}
        />

        {/* Section 3: Users */}
        <UsersSection />


      </main>

      {/* Footer */}
      <footer className="bg-black mt-20 py-8">
        <div className="container mx-auto px-4 text-center text-slate-400">
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
