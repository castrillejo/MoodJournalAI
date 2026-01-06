import { useState } from 'react';
import { Send, Sparkles } from 'lucide-react';
import { motion } from 'framer-motion';

const TextInput = ({ onAnalyze, isLoading, showAttention, setShowAttention }) => {
    const [text, setText] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (text.trim()) onAnalyze(text);
    };

    const examples = [
        { emotion: 'joy', text: 'I feel so happy and excited today!' },
        { emotion: 'sadness', text: 'I am feeling really sad and lonely right now' },
        { emotion: 'fear', text: 'I am terrified about what might happen tomorrow' }
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="w-full max-w-4xl mx-auto"
        >
            <form onSubmit={handleSubmit} className="space-y-4">
                <div className="relative">
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        placeholder="Write your text in English here... (e.g., I feel amazing today!)"
                        className="w-full px-6 py-4 text-lg rounded-2xl resize-none transition-all duration-200
                       bg-slate-900/60 border border-slate-700 text-slate-100 placeholder:text-slate-500
                       focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500
                       shadow-sm shadow-black/20"
                        rows="5"
                        disabled={isLoading}
                    />
                    <div className="absolute bottom-4 right-4 text-sm text-slate-500">
                        {text.length} / 512
                    </div>
                </div>

                <div className="flex items-center justify-between flex-wrap gap-4">
                    <label className="flex items-center space-x-2 cursor-pointer group">
                        <input
                            type="checkbox"
                            checked={showAttention}
                            onChange={(e) => setShowAttention(e.target.checked)}
                            className="w-5 h-5 text-purple-500 rounded focus:ring-purple-500"
                            disabled={isLoading}
                        />
                        <span className="text-sm font-medium text-slate-300 group-hover:text-purple-300 transition-colors flex items-center gap-2">
                            <Sparkles className="w-4 h-4" />
                            Show Attention Weights
                        </span>
                    </label>

                    <motion.button
                        type="submit"
                        disabled={!text.trim() || isLoading}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className={`px-8 py-3 rounded-xl font-semibold text-white transition-all duration-200 flex items-center gap-2
                        shadow-lg shadow-black/30 border border-slate-800
                        ${!text.trim() || isLoading
                                ? 'bg-slate-700/60 cursor-not-allowed'
                                : showAttention
                                    ? 'bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700'
                                    : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700'
                            }`}
                    >
                        {isLoading ? (
                            <>
                                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                                Analyzing...
                            </>
                        ) : (
                            <>
                                <Send className="w-5 h-5" />
                                {showAttention ? 'Analyze with Attention' : 'Analyze Emotion'}
                            </>
                        )}
                    </motion.button>
                </div>
            </form>

            {/* Quick Examples */}
            <div className="mt-6">
                <p className="text-sm font-medium text-slate-400 mb-3">💡 Quick Examples:</p>
                <div className="flex flex-wrap gap-2">
                    {examples.map((example, index) => (
                        <motion.button
                            key={index}
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => setText(example.text)}
                            className="px-4 py-2 text-sm rounded-lg transition-all duration-200
                         bg-slate-900/70 border border-slate-700 text-slate-200
                         hover:border-purple-500 hover:text-purple-300
                         shadow-sm shadow-black/20"
                            disabled={isLoading}
                        >
                            {example.text.substring(0, 30)}...
                        </motion.button>
                    ))}
                </div>
            </div>
        </motion.div>
    );
};

export default TextInput;
