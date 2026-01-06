import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid } from 'recharts';
import { EMOTION_COLORS, EMOTION_EMOJIS } from '../constants/emotions';
import { motion } from 'framer-motion';

const EmotionChart = ({ scores, type }) => {
    const chartData = scores
        .map(({ emotion, score }) => ({
            name: `${EMOTION_EMOJIS[emotion]} ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`,
            value: score * 100,
            emotion,
            color: EMOTION_COLORS[emotion],
        }))
        .sort((a, b) => b.value - a.value);

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-slate-950/90 px-4 py-2 rounded-lg border border-slate-800 shadow-xl shadow-black/40">
                    <p className="font-semibold text-slate-100">{payload[0].payload.name}</p>
                    <p className="text-sm" style={{ color: payload[0].payload.color }}>
                        {payload[0].value.toFixed(1)}%
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl border border-slate-800 bg-slate-900/60 shadow-lg shadow-black/20 p-6"
        >
            <h3 className="text-xl font-bold text-slate-100 mb-4">
                {type === 'bar' ? '📊 Emotion Distribution' : '🥧 Top 3 Emotions'}
            </h3>

            <ResponsiveContainer width="100%" height={300}>
                {type === 'bar' ? (
                    <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16 }}>
                        <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
                        <XAxis
                            type="number"
                            domain={[0, 100]}
                            tickFormatter={(v) => `${v}%`}
                            tick={{ fill: '#cbd5e1', fontSize: 12 }}
                            axisLine={{ stroke: '#334155' }}
                            tickLine={{ stroke: '#334155' }}
                        />
                        <YAxis
                            dataKey="name"
                            type="category"
                            width={150}
                            tick={{ fill: '#cbd5e1', fontSize: 12 }}
                            axisLine={{ stroke: '#334155' }}
                            tickLine={{ stroke: '#334155' }}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                            {chartData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Bar>
                    </BarChart>
                ) : (
                    <PieChart>
                        <Tooltip content={<CustomTooltip />} />
                        <Pie
                            data={chartData.slice(0, 3)}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={100}
                            dataKey="value"
                            labelLine={false}
                        >
                            {chartData.slice(0, 3).map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Pie>
                        <Legend
                            verticalAlign="bottom"
                            height={36}
                            formatter={(value) => <span className="text-slate-200">{value}</span>}
                        />
                    </PieChart>
                )}
            </ResponsiveContainer>
        </motion.div>
    );
};

export default EmotionChart;
