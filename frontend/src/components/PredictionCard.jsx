import { motion } from 'framer-motion';
import { EMOTION_EMOJIS, EMOTION_NAMES_ES, EMOTION_COLORS } from '../constants/emotions';
import EmotionChart from './EmotionChart';

const pct = (x) => (typeof x === 'number' ? `${(x * 100).toFixed(1)}%` : '-');

const PredictionCard = ({ title, result }) => {
    if (!result) return null;

    const { predicted_emotion, confidence, all_scores } = result;
    const emoji = EMOTION_EMOJIS[predicted_emotion] || '🎭';
    const es = EMOTION_NAMES_ES[predicted_emotion] || '';
    const color = EMOTION_COLORS[predicted_emotion] || '#7c3aed';

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100"
        >
            <div className="flex items-start justify-between gap-3">
                <div>
                    <p className="text-sm font-semibold text-gray-500">{title}</p>
                    <div className="mt-2 flex items-center gap-3">
                        <span className="text-4xl">{emoji}</span>
                        <div>
                            <div className="text-2xl font-black uppercase" style={{ color }}>
                                {predicted_emotion}
                            </div>
                            <div className="text-sm text-gray-500">({es})</div>
                        </div>
                    </div>
                </div>

                <div className="text-right">
                    <div className="text-xs text-gray-500">Confianza</div>
                    <div className="text-lg font-bold text-gray-800">{pct(confidence)}</div>
                </div>
            </div>

            <div className="mt-5">
                <EmotionChart scores={all_scores || []} type="pie" />
            </div>
        </motion.div>
    );
};

export default PredictionCard;
