import { motion } from 'framer-motion';

const SemiFrozenSelector = ({ selected, onChange }) => {
    const options = [
        { key: 'semi_frozen2', label: '2 capas' },
        { key: 'semi_frozen4', label: '4 capas' },
        { key: 'semi_frozen6', label: '6 capas' },
    ];

    return (
        <div className="flex items-center gap-2 flex-wrap">
            {options.map((opt) => {
                const active = selected === opt.key;
                return (
                    <motion.button
                        key={opt.key}
                        whileHover={{ scale: 1.03 }}
                        whileTap={{ scale: 0.97 }}
                        onClick={() => onChange(opt.key)}
                        className={[
                            'px-4 py-2 rounded-full text-sm font-semibold border transition',
                            active
                                ? 'bg-purple-600 text-white border-purple-600 shadow'
                                : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50',
                        ].join(' ')}
                        type="button"
                    >
                        {opt.label}
                    </motion.button>
                );
            })}
        </div>
    );
};

export default SemiFrozenSelector;