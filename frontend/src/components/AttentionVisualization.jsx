import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
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

    const hexToRgb = (hex) => {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result
            ? {
                r: parseInt(result[1], 16),
                g: parseInt(result[2], 16),
                b: parseInt(result[3], 16),
            }
            : null;
    };

    const rgb = hexToRgb(emotionColor);

    const readableTextColor = (() => {
        if (!rgb) return '#f8fafc';
        const luminance = 0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b;
        return luminance > 160 ? '#0b1220' : '#f8fafc';
    })();

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-slate-800 bg-slate-900/60 shadow-lg shadow-black/20 p-6 space-y-6"
        >
            <div>
                <h3 className="text-2xl font-bold text-slate-100 mb-2">
                    🔍 Attention Weights Visualization
                </h3>
                <p className="text-slate-300">
                    Words highlighted below had the most influence on the prediction.
                </p>
            </div>

            {/* Highlighted Text */}
            <div className="rounded-xl p-6 border border-slate-800 bg-slate-950/50">
                <p className="text-sm font-medium text-slate-400 mb-3">
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
                                initial={{ opacity: 0, scale: 0.8 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: index * 0.05 }}
                                className="px-2 py-1 rounded"
                                style={{
                                    backgroundColor: rgb
                                        ? `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`
                                        : `rgba(148, 163, 184, ${alpha})`,
                                    fontWeight,
                                    color: readableTextColor,
                                }}
                            >
                                {token}
                            </motion.span>
                        );
                    })}
                </div>
            </div>

            {/* Charts Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Bar Chart */}
                <div>
                    <h4 className="text-lg font-semibold text-slate-200 mb-3">
                        📊 Top 10 Words by Attention
                    </h4>

                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16 }}>
                            <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                            <XAxis
                                type="number"
                                domain={[0, 100]}
                                tickFormatter={(v) => `${v.toFixed(0)}%`}
                                tick={{ fill: '#cbd5e1', fontSize: 12 }}
                                axisLine={{ stroke: '#334155' }}
                                tickLine={{ stroke: '#334155' }}
                            />
                            <YAxis
                                dataKey="token"
                                type="category"
                                width={90}
                                tick={{ fill: '#cbd5e1', fontSize: 12 }}
                                axisLine={{ stroke: '#334155' }}
                                tickLine={{ stroke: '#334155' }}
                            />
                            <Tooltip
                                formatter={(value) => `${value.toFixed(1)}%`}
                                contentStyle={{ backgroundColor: 'rgba(2,6,23,0.92)', border: '1px solid #334155', borderRadius: '12px' }}
                                labelStyle={{ color: '#e2e8f0' }}
                                itemStyle={{ color: '#e2e8f0' }}
                            />
                            <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                                {chartData.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={emotionColor}
                                        opacity={0.6 + (entry.score / 100) * 0.4}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                {/* Top 5 List */}
                <div>
                    <h4 className="text-lg font-semibold text-slate-200 mb-3">
                        📌 Top 5 Key Words
                    </h4>

                    <div className="space-y-3">
                        {topTokens.map((item, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className="flex items-center justify-between p-3 rounded-lg border border-slate-800 bg-slate-950/40"
                            >
                                <div className="flex items-center gap-3">
                                    <span className="flex items-center justify-center w-8 h-8 rounded-full border border-purple-900 bg-purple-950 text-purple-200 font-bold text-sm">
                                        {index + 1}
                                    </span>
                                    <code className="text-lg font-mono font-semibold text-slate-100">
                                        {item.token}
                                    </code>
                                </div>

                                <span className="text-sm font-semibold" style={{ color: emotionColor }}>
                                    {item.score.toFixed(1)}%
                                </span>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default AttentionVisualization;
