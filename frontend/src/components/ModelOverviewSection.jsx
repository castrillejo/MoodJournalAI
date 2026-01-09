import { motion } from "framer-motion";
import ConfusionMatrixGrid from "./ConfusionMatrixGrid";
import MetricsPanel from "./MetricsPanel";
import SemiFrozenSelector from "./SemiFrozenSelector";

const ModelOverviewSection = ({ overview, selectedSemi, setSelectedSemi }) => {
    const models = overview?.models || {};

    const fin = models?.finetuned || {};
    const frozen = models?.frozen || {};

    const semiVariants = models?.semi_frozen?.variants || {};
    const semi = semiVariants?.[selectedSemi] || null;

    const cmLabels = overview?.confusion_matrix?.labels || [];
    const cmMatrix = overview?.confusion_matrix?.matrix || [];

    return (
        <section id="overview" className="scroll-mt-28">
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="max-w-6xl mx-auto mb-6">
                    <h2 className="text-3xl font-bold text-gray-900">Evaluacion del Model</h2>
                    <p className="text-gray-600 mt-1">
                        Metricas de evaluacion para el modelo fine-tuned basado en RoBERTa.
                    </p>
                </div>

                {/* 1) Confusion matrix (full width) */}
                <div className="max-w-6xl mx-auto">
                    <ConfusionMatrixGrid labels={cmLabels} matrix={cmMatrix} />
                </div>

                {/* 2) Fine-tuned metrics (full width) */}
                <div className="max-w-6xl mx-auto mt-6">
                    <MetricsPanel
                        title="Fine-tuned (modelo completamente entrenado)"
                        metrics={fin?.metrics}
                        perClass={fin?.per_class}
                    />
                </div>

                {/* 3) Frozen + Semi-frozen (2 columns) */}
                <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
                    <MetricsPanel
                        title="Congelado"
                        subtitle="Encoder congelado, clasificador entrenado"
                        metrics={frozen?.metrics}
                        perClass={frozen?.per_class}
                        compact
                    />

                    <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
                        <div className="flex items-start justify-between flex-wrap gap-4">
                            <div>
                                <h3 className="text-lg font-bold text-gray-900">Semicongelado</h3>
                                <p className="text-sm text-gray-600 mt-1">
                                    Cambia entre los modelos semicongelados con 2, 4 y 6 capas descongeladas.
                                </p>
                            </div>

                            <SemiFrozenSelector selected={selectedSemi} onChange={setSelectedSemi} />
                        </div>

                        <div className="mt-5">
                            {semi ? (
                                <MetricsPanel
                                    title={`Modelo seleccionado: ${selectedSemi}`}
                                    metrics={semi?.metrics}
                                    perClass={semi?.per_class}
                                    compact
                                />
                            ) : (
                                <p className="text-sm text-gray-600">
                                    No se ha encontrado datos para el modelo <b>{selectedSemi}</b>.
                                </p>
                            )}
                        </div>
                    </div>
                </div>
            </motion.div>
        </section>
    );
};

export default ModelOverviewSection;
