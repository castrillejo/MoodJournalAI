const ConfusionMatrixGrid = ({ labels, matrix }) => {
    if (!labels?.length || !matrix?.length) return null;

    const flat = matrix.flat();
    const max = Math.max(...flat, 1);

    const cellStyle = (v) => {
        const a = Math.min(0.85, 0.08 + (v / max) * 0.77);
        return { backgroundColor: `rgba(168, 85, 247, ${a})` }; // purple-500-ish
    };

    return (
        <div className="bg-white rounded-2xl shadow-lg p-6 border border-gray-100 overflow-auto">
            <h3 className="text-lg font-bold text-gray-800 mb-4">Confusion Matrix (Fine-tuned)</h3>

            <div className="min-w-[720px]">
                <table className="w-full border-separate border-spacing-2">
                    <thead>
                        <tr>
                            <th className="text-left text-xs text-gray-500 p-2">Real \ Pred</th>
                            {labels.map((l) => (
                                <th key={l} className="text-xs text-gray-600 p-2 text-center">
                                    {l}
                                </th>
                            ))}
                        </tr>
                    </thead>

                    <tbody>
                        {labels.map((rowLabel, i) => (
                            <tr key={rowLabel}>
                                <th className="text-xs text-gray-600 p-2 text-left">{rowLabel}</th>
                                {labels.map((colLabel, j) => {
                                    const v = matrix[i]?.[j] ?? 0;
                                    return (
                                        <td
                                            key={`${rowLabel}-${colLabel}`}
                                            className="p-2 rounded-xl text-center font-semibold text-white"
                                            style={cellStyle(v)}
                                            title={`${rowLabel} → ${colLabel}: ${v}`}
                                        >
                                            {v}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default ConfusionMatrixGrid;