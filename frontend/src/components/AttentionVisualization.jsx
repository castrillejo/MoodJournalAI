import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { EMOTION_COLORS } from '../constants/emotions';

const AttentionVisualization = ({ attention, emotion }) => {
    const { tokens, scores } = attention;
    const emotionColor = EMOTION_COLORS[emotion];

    const chartData = tokens
        .map((token, index) => ({
            token: token.trim(),
            score: scores[index] * 100,
        }))
        .sort((a, b) => b.score - a.score)
        .slice(0, 10);

    const topTokens = [...chartData].slice(0, 5);

    return (
        <div className="space-y-5">
            {/* Highlighted Text */}
            <div className="bg-slate-950/40 rounded-xl p-5 border border-slate-800">
                <p className="text-sm font-medium text-slate-300 mb-3">
                    💡 Highlighted Text (darker = more important)
                </p>
                <div className="text-lg leading-relaxed flex flex-wrap gap-1">
                    {tokens.map((token, index) => {
                        const score = scores[index];
                        const alpha = score * 0.7;
                        const fontWeight = 400 + Math.floor(score * 300);

                        return (
                            <motion.span
                                key={index}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: index * 0.01 }}
                                className="px-1 rounded"
                                style={{
                                    backgroundColor: `rgba(168, 85, 247, ${alpha})`,
                                    fontWeight,
                                    color: 'rgba(255,255,255,0.95)',
                                }}
                            >
                                {token}
                            </motion.span>
                        );
                    })}
                </div>
            </div>

            {/* Top 5 */}
            <div className="bg-slate-950/30 rounded-xl p-5 border border-slate-800">
                <p className="text-sm font-semibold text-slate-200 mb-3">🏆 Top 5 tokens</p>
                <div className="space-y-2">
                    {topTokens.map((t, i) => (
                        <div key={i} className="flex items-center justify-between text-sm">
                            <span className="font-semibold text-slate-100">{t.token || '(blank)'}</span>
                            <span className="text-slate-300">{t.score.toFixed(2)}%</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Histogram */}
            <div className="bg-slate-950/30 rounded-xl p-5 border border-slate-800">
                <p className="text-sm font-semibold text-slate-200 mb-3">📊 Top 10 attention histogram</p>
                <div className="h-52">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData}>
                            <XAxis dataKey="token" tick={{ fill: '#cbd5e1', fontSize: 11 }} interval={0} angle={-20} textAnchor="end" height={60} />
                            <YAxis tick={{ fill: '#cbd5e1', fontSize: 11 }} />
                            <Tooltip
                                contentStyle={{ background: '#0b1220', border: '1px solid #334155', color: '#e2e8f0' }}
                                labelStyle={{ color: '#e2e8f0' }}
                            />
                            <Bar dataKey="score">
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={emotionColor} opacity={0.6 + (index * 0.04)} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default AttentionVisualization;
