import { useState } from 'react';
import { motion } from 'framer-motion';

import TextInput from './TextInput';
import PredictionCard from './PredictionCard';
import AttentionVisualization from './AttentionVisualization';

const ComparePredictionsSection = ({
    selectedSemi,
    onAnalyzeCompare,
    compareLoading,
    compareError,
    compareResult,
}) => {
    const frozen = compareResult?.frozen || null;
    const semi = compareResult?.semi || null;
    const finetuned = compareResult?.finetuned || null;

    return (
        <section id="compare" className="scroll-mt-28 mt-12">
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="max-w-6xl mx-auto mb-6">
                    <h2 className="text-3xl font-bold text-gray-800">Comparar Prediccion Puntual</h2>
                    <p className="text-gray-600 mt-1">
                        Ejecutamos el mismo input a través del modelo completamente congelado, el semicongelado ({selectedSemi}) y el fine-tuned.
                    </p>
                </div>

                <TextInput onAnalyze={onAnalyzeCompare} isLoading={compareLoading} />

                {compareError && (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.98 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className="max-w-5xl mx-auto mt-6 p-5 bg-red-50 border border-red-200 rounded-2xl"
                    >
                        <p className="text-red-800 text-center font-medium">{compareError}</p>
                    </motion.div>
                )}

                {(frozen || semi || finetuned) && (
                    <div className="max-w-6xl mx-auto mt-8 space-y-6">
                        {/* 3 pie charts aligned */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <PredictionCard title="Clasificador congelado" result={frozen} />
                            <PredictionCard title={`Clasificador semicongelado (${selectedSemi})`} result={semi} />
                            <PredictionCard title="Clasificador fine-tuned" result={finetuned} />
                        </div>

                        {/* Attention blocks */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                            <div className="bg-white rounded-2xl shadow-lg p-5 border border-gray-100">
                                <p className="text-sm font-semibold text-gray-600 mb-3">Attention — Congelado</p>
                                {frozen?.attention ? (
                                    <AttentionVisualization attention={frozen.attention} emotion={frozen.predicted_emotion} />
                                ) : (
                                    <p className="text-sm text-gray-500">No attention data.</p>
                                )}
                            </div>

                            <div className="bg-white rounded-2xl shadow-lg p-5 border border-gray-100">
                                <p className="text-sm font-semibold text-gray-600 mb-3">Attention — Semicongelado</p>
                                {semi?.attention ? (
                                    <AttentionVisualization attention={semi.attention} emotion={semi.predicted_emotion} />
                                ) : (
                                    <p className="text-sm text-gray-500">No attention data.</p>
                                )}
                            </div>

                            <div className="bg-white rounded-2xl shadow-lg p-5 border border-gray-100">
                                <p className="text-sm font-semibold text-gray-600 mb-3">Attention — Fine-tuned</p>
                                {finetuned?.attention ? (
                                    <AttentionVisualization attention={finetuned.attention} emotion={finetuned.predicted_emotion} />
                                ) : (
                                    <p className="text-sm text-gray-500">No attention data.</p>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </motion.div>
        </section>
    );
};

export default ComparePredictionsSection;
