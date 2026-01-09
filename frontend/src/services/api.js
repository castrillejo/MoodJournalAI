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
    try {
        const response = await api.post('/predict/compare/attention', {
            text,
            semi_variant: semiVariantKey,
        });
        return response.data;
    } catch (err) {
        const status = err?.response?.status;
        if (status !== 404) {
            throw err;
        }
    }

    const [frozen, semi, finetuned] = await Promise.all([
        predictEmotionWithAttention(text, 'frozen'),
        predictEmotionWithAttention(text, semiVariantKey),
        predictEmotionWithAttention(text, 'finetuned'),
    ]);

    return { frozen, semi, finetuned };
};

export const searchUsers = async (q, limit = 5) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/users/search`, {
            params: { q, limit },
        });
        return response.data;
    } catch (error) {
        console.error('Error searching users:', error);
        throw error;
    }
};

export const getUserStats = async (id) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/users/${id}/stats`);
        return response.data;
    } catch (error) {
        console.error('Error fetching user stats:', error);
        throw error;
    }
};