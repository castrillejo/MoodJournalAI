import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export const predictEmotion = async (text) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/predict`, {
            text: text
        });
        return response.data;
    } catch (error) {
        console.error('Error predicting emotion:', error);
        throw error;
    }
};

export const predictEmotionWithAttention = async (text) => {
    try {
        const response = await axios.post(`${API_BASE_URL}/predict/attention`, {
            text: text
        });
        return response.data;
    } catch (error) {
        console.error('Error predicting with attention:', error);
        throw error;
    }
};

export const checkAPIHealth = async () => {
    try {
        const response = await axios.get('http://localhost:8000/health');
        return response.data;
    } catch (error) {
        return { status: 'offline' };
    }
};
