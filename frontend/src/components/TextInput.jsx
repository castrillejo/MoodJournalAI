import { useMemo, useState } from 'react';
import { Send, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const TextInput = ({ onAnalyze, isLoading }) => {
    const [text, setText] = useState('');

    const examples = useMemo(() => ([
        { emotion: 'joy', text: 'I feel so happy and excited today!' },
        { emotion: 'sadness', text: 'I feel really down and empty this evening.' },
        { emotion: 'fear', text: 'I am nervous and scared about what might happen.' },
        { emotion: 'anger', text: 'I am furious about how unfair this is.' },
        { emotion: 'love', text: 'I feel deeply grateful and connected to you.' },
        { emotion: 'surprise', text: 'I did not expect that at all, wow!' },
    ]), []);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!text.trim() || isLoading) return;
        onAnalyze(text);
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-5xl mx-auto"
        >
            <div className="bg-white rounded-3xl shadow-xl p-8 border border-gray-100">
                <div className="flex items-center gap-3 mb-6">
                    <div className="p-3 bg-purple-100 rounded-2xl">
                        <Sparkles className="w-6 h-6 text-purple-600" />
                    </div>
                    <div>
                        <h2 className="text-2xl font-bold text-gray-800">Escribe una breve entrada de prueba</h2>
                    </div>
                </div>

                <form onSubmit={handleSubmit} className="space-y-5">
                    <div className="relative">
                        <textarea
                            value={text}
                            onChange={(e) => setText(e.target.value.slice(0, 512))}
                            placeholder="Escribe tu texto en inglés aquí... (e.g., I feel amazing today!)"
                            className="w-full px-6 py-4 text-lg border-2 border-gray-200 rounded-2xl focus:border-purple-400 focus:ring-4 focus:ring-purple-100 outline-none resize-none transition-all duration-200 shadow-sm hover:shadow-md"
                            rows="5"
                            disabled={isLoading}
                        />
                        <div className="absolute bottom-4 right-4 text-sm text-gray-400">
                            {text.length} / 512
                        </div>
                    </div>

                    <div className="flex items-center justify-between flex-wrap gap-4">
                        <p className="text-sm text-gray-500">

                        </p>

                        <motion.button
                            whileHover={{ scale: 1.03 }}
                            whileTap={{ scale: 0.97 }}
                            type="submit"
                            disabled={isLoading || !text.trim()}
                            className="px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-2xl font-semibold shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-3"
                        >
                            {isLoading ? (
                                <>
                                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                                    Analizando...
                                </>
                            ) : (
                                <>
                                    <Send className="w-5 h-5" />
                                    Comparando Predicciones
                                </>
                            )}
                        </motion.button>
                    </div>
                </form>

                {/* Quick Examples */}
                <div className="mt-7">
                    <p className="text-sm font-medium text-gray-600 mb-3">💡 Ejemplos rápidos:</p>
                    <div className="flex flex-wrap gap-2">
                        {examples.map((ex, idx) => (
                            <motion.button
                                key={idx}
                                whileHover={{ y: -2 }}
                                whileTap={{ scale: 0.97 }}
                                onClick={() => setText(ex.text)}
                                className="px-4 py-2 text-sm bg-gray-50 border border-gray-200 rounded-full text-gray-700 hover:border-purple-300 hover:text-purple-600 transition-all duration-200 shadow-sm"
                                disabled={isLoading}
                            >
                                {ex.text.substring(0, 30)}...
                            </motion.button>
                        ))}
                    </div>
                </div>
            </div>
        </motion.div>
    );
};

export default TextInput;