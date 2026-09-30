import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_URL,
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const getModels = async () => {
  const res = await api.get('/models');
  return res.data;
};

export const predict = async (imageFile, modelKey) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('model', modelKey);
  const res = await api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
};

export const getMetrics = async () => {
  const res = await api.get('/metrics');
  return res.data;
};

export const compareModels = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  const res = await api.post('/compare', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return res.data;
};
