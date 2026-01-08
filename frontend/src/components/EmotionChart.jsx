import {
    BarChart,
    Bar,
    PieChart,
    Pie,
    Cell,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
} from "recharts";
import { EMOTION_COLORS, EMOTION_EMOJIS } from "../constants/emotions";
import { motion } from "framer-motion";

const EmotionChart = ({ scores, type }) => {
    const chartData = scores
        .map(({ emotion, score }) => ({
            name: `${EMOTION_EMOJIS[emotion]} ${emotion.charAt(0).toUpperCase() + emotion.slice(1)}`,
            value: score * 100,
            emotion,
            color: EMOTION_COLORS[emotion],
        }))
        .sort((a, b) => b.value - a.value);

    const top3 = chartData.slice(0, 3);

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-white px-4 py-2 rounded-lg shadow-lg border-2 border-gray-200">
                    <p className="font-semibold">{payload[0].payload.name}</p>
                    <p className="text-sm" style={{ color: payload[0].payload.color }}>
                        {payload[0].value.toFixed(1)}%
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="bg-white rounded-2xl shadow-lg p-6">
            <h3 className="text-xl font-bold text-gray-800 mb-4">
                {type === "bar" ? "📊 Emotion Distribution" : "🥧 Top 3 Emotions"}
            </h3>

            <ResponsiveContainer width="100%" height={300}>
                {type === "bar" ? (
                    <BarChart data={chartData} layout="vertical">
                        <XAxis type="number" domain={[0, 100]} tickFormatter={(value) => `${value}%`} />
                        <YAxis dataKey="name" type="category" width={120} />
                        <Tooltip content={<CustomTooltip />} />
                        <Bar dataKey="value" radius={[0, 8, 8, 0]}>
                            {chartData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Bar>
                    </BarChart>
                ) : (
                    <PieChart>
                        <Pie
                            data={top3}
                            cx="50%"
                            cy="50%"
                            innerRadius={60}
                            outerRadius={100}
                            dataKey="value"
                            label={false}
                        >
                            {top3.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                        </Pie>
                        <Tooltip content={<CustomTooltip />} />
                    </PieChart>
                )}
            </ResponsiveContainer>

            {/* Leyenda custom (solo pie): 1 por línea, orden descendente */}
            {type !== "bar" && (
                <div className="mt-4 flex flex-col items-start gap-2">
                    {top3.map((entry) => (
                        <div key={entry.emotion} className="flex items-center gap-2">
                            <span className="inline-block w-3 h-3 rounded-sm" style={{ background: entry.color }} />
                            <span>{EMOTION_EMOJIS[entry.emotion]}</span>
                            <span className="font-medium text-gray-800">
                                {entry.emotion.charAt(0).toUpperCase() + entry.emotion.slice(1)}
                            </span>
                        </div>
                    ))}
                </div>
            )}
        </motion.div>
    );
};

export default EmotionChart;
