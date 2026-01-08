import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: 60000,
});

export const checkAPIHealth = async () => {
    try {
        const response = await axios.get('http://localhost:8000/health');
        return response.data;
    } catch (error) {
        return { status: 'offline' };
    }
};

export const fetchEvaluationOverview = async () => {
    const response = await api.get('/evaluation/overview');
    return response.data;
};

export const predictEmotionWithAttention = async (text, model) => {
    const payload = { text };
    if (model) payload.model = model;

    const response = await api.post('/predict/attention', payload);
    return response.data;
};

export const predictCompareWithAttention = async (text, semiVariantKey) => {
    // 1) Intento endpoint dedicado (si lo implementas luego)
    try {
        const response = await api.post('/predict/compare/attention', {
            text,
            semi_variant: semiVariantKey,
        });
        return response.data; // esperado: { frozen: {...}, semi: {...}, finetuned: {...} }
    } catch (err) {
        const status = err?.response?.status;
        if (status !== 404) {
            // No era "no existe", así que lanzamos error real
            throw err;
        }
        // 2) Fallback: 3 llamadas
    }

    const [frozen, semi, finetuned] = await Promise.all([
        predictEmotionWithAttention(text, 'frozen'),
        predictEmotionWithAttention(text, semiVariantKey),
        predictEmotionWithAttention(text, 'finetuned'),
    ]);

    return { frozen, semi, finetuned };
};