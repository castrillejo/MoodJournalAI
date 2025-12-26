import { motion } from 'framer-motion';
import { EMOTION_EMOJIS, EMOTION_COLORS, EMOTION_NAMES_ES } from '../constants/emotions';
import EmotionChart from './EmotionChart';
import AttentionVisualization from './AttentionVisualization';

const ResultCard = ({ result, showAttention }) => {
    const { predicted_emotion, confidence, all_scores, attention } = result;
    const emoji = EMOTION_EMOJIS[predicted_emotion];
    const color = EMOTION_COLORS[predicted_emotion];
    const nameEs = EMOTION_NAMES_ES[predicted_emotion];

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-6xl mx-auto mt-8 space-y-6"
        >
            {/* Main Result Card */}
            <motion.div
                initial={{ y: 20 }}
                animate={{ y: 0 }}
                className="bg-white rounded-3xl shadow-2xl overflow-hidden"
                style={{
                    background: `linear-gradient(135deg, ${color}15 0%, ${color}30 100%)`
                }}
            >
                <div className="p-8 text-center">
                    <motion.div
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ type: 'spring', stiffness: 200, damping: 10 }}
                        className="text-9xl mb-4"
                    >
                        {emoji}
                    </motion.div>
                    <h2
                        className="text-5xl font-bold uppercase mb-2"
                        style={{ color }}
                    >
                        {predicted_emotion}
                    </h2>
                    <p className="text-2xl text-gray-600 mb-6">
                        ({nameEs})
                    </p>
                    <div className="inline-block px-6 py-3 bg-white rounded-full shadow-md">
                        <p className="text-lg font-semibold text-gray-700">
                            Confidence: <span style={{ color }}>{(confidence * 100).toFixed(1)}%</span>
                        </p>
                    </div>
                </div>
            </motion.div>

            {/* Attention Visualization */}
            {showAttention && attention && (
                <AttentionVisualization
                    attention={attention}
                    emotion={predicted_emotion}
                />
            )}

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <EmotionChart scores={all_scores} type="bar" />
                <EmotionChart scores={all_scores} type="pie" />
            </div>

            {/* Details */}
            <motion.details
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-white rounded-2xl shadow-lg p-6"
            >
                <summary className="cursor-pointer font-semibold text-gray-700 hover:text-purple-600 transition-colors">
                    📋 View Complete Details
                </summary>
                <pre className="mt-4 p-4 bg-gray-50 rounded-lg overflow-auto text-sm">
                    {JSON.stringify(result, null, 2)}
                </pre>
            </motion.details>
        </motion.div>
    );
};

export default ResultCard;
