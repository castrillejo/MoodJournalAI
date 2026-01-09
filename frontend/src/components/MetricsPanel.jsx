const pct = (x) => (typeof x === 'number' ? `${(x * 100).toFixed(1)}%` : '-');

const MetricsPanel = ({ title, subtitle, metrics, perClass, compact = false }) => {
    return (
        <div className="bg-slate-900/60 rounded-2xl shadow-xl p-6 border border-slate-800">
            <div className="mb-4">
                <h3 className="text-lg font-bold text-slate-100">{title}</h3>
                {subtitle && <p className="text-sm text-slate-400 mt-1">{subtitle}</p>}
            </div>

            <div className="space-y-1 text-sm">
                <div className="flex items-center justify-between">
                    <span className="text-slate-400">Accuracy</span>
                    <span className="font-semibold text-slate-100">{pct(metrics?.accuracy)}</span>
                </div>
                <div className="flex items-center justify-between">
                    <span className="text-slate-400">Precision (ponderada)</span>
                    <span className="font-semibold text-slate-100">{pct(metrics?.precision_weighted)}</span>
                </div>
                <div className="flex items-center justify-between">
                    <span className="text-slate-400">Recall (ponderada)</span>
                    <span className="font-semibold text-slate-100">{pct(metrics?.recall_weighted)}</span>
                </div>
                <div className="flex items-center justify-between">
                    <span className="text-slate-400">F1 (ponderada)</span>
                    <span className="font-semibold text-slate-100">{pct(metrics?.f1_weighted)}</span>
                </div>
            </div>

            {!compact && perClass && (
                <div className="mt-5">
                    <p className="text-sm font-semibold text-slate-200 mb-2">Por clase</p>
                    <div className="overflow-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="text-left text-slate-400">
                                    <th className="py-2 pr-3">Clase</th>
                                    <th className="py-2 pr-3">Precision</th>
                                    <th className="py-2 pr-3">Recall</th>
                                    <th className="py-2 pr-3">F1</th>
                                </tr>
                            </thead>
                            <tbody>
                                {Object.entries(perClass).map(([label, vals]) => (
                                    <tr key={label} className="border-t border-slate-800">
                                        <td className="py-2 pr-3 font-semibold text-slate-100">{label}</td>
                                        <td className="py-2 pr-3 text-slate-200">{pct(vals?.precision)}</td>
                                        <td className="py-2 pr-3 text-slate-200">{pct(vals?.recall)}</td>
                                        <td className="py-2 pr-3 text-slate-200">{pct(vals?.f1)}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MetricsPanel;
