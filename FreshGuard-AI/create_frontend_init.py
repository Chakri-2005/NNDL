import os
import shutil

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

frontend_src = r"C:\NNDL\FreshGuard-AI\frontend\src"

# Dirs to create
dirs = [
    "components",
    "pages",
    "services",
    "hooks",
    "context",
    "utils",
    "styles",
]
for d in dirs:
    os.makedirs(os.path.join(frontend_src, d), exist_ok=True)

# 1. services/api.js
write_file(os.path.join(frontend_src, "services", "api.js"), """import axios from 'axios';

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
""")

# 2. styles/variables.css
write_file(os.path.join(frontend_src, "styles", "variables.css"), """:root {
  --primary: #0ea5e9;
  --primary-dark: #0284c7;
  --primary-light: #e0f2fe;
  
  --success: #10b981;
  --success-dark: #059669;
  --success-light: #d1fae5;
  
  --danger: #ef4444;
  --danger-dark: #dc2626;
  --danger-light: #fee2e2;
  
  --warning: #f59e0b;
  --warning-dark: #d97706;
  --warning-light: #fef3c7;
  
  --bg-color: #f8fafc;
  --card-bg: #ffffff;
  
  --text-main: #0f172a;
  --text-muted: #64748b;
  
  --border: #e2e8f0;
  
  --radius-sm: 0.375rem;
  --radius: 0.5rem;
  --radius-lg: 1rem;
  --radius-xl: 1.5rem;
  
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  --shadow-hover: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
  
  --font-sans: 'Inter', system-ui, -apple-system, sans-serif;
}
""")

# 3. styles/global.css
write_file(os.path.join(frontend_src, "styles", "global.css"), """@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
@import './variables.css';

* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: var(--font-sans);
  background-color: var(--bg-color);
  color: var(--text-main);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

a {
  text-decoration: none;
  color: inherit;
}

button {
  cursor: pointer;
  font-family: inherit;
  border: none;
  outline: none;
  transition: all 0.2s ease;
}

img {
  max-width: 100%;
  display: block;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1.5rem;
}

/* Animations */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

.animate-fade-in {
  animation: fadeIn 0.4s ease-out forwards;
}
""")

# 4. components/Navbar.jsx
write_file(os.path.join(frontend_src, "components", "Navbar.jsx"), """import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Leaf, Camera } from 'lucide-react';
import '../styles/navbar.css';

const Navbar = () => {
  const location = useLocation();
  const isActive = (path) => location.pathname === path;

  return (
    <nav className="navbar">
      <div className="navbar-container container">
        <Link to="/" className="navbar-logo">
          <Leaf className="logo-icon" />
          <span>FreshGuard AI</span>
        </Link>
        
        <div className="navbar-links">
          <Link to="/" className={`nav-link ${isActive('/') ? 'active' : ''}`}>Home</Link>
          <Link to="/compare" className={`nav-link ${isActive('/compare') ? 'active' : ''}`}>Compare</Link>
          <Link to="/dashboard" className={`nav-link ${isActive('/dashboard') ? 'active' : ''}`}>Dashboard</Link>
          <Link to="/history" className={`nav-link ${isActive('/history') ? 'active' : ''}`}>History</Link>
          <Link to="/research" className={`nav-link ${isActive('/research') ? 'active' : ''}`}>Research</Link>
          <Link to="/about" className={`nav-link ${isActive('/about') ? 'active' : ''}`}>About</Link>
        </div>
        
        <Link to="/scan" className="btn-primary btn-scan">
          <Camera size={18} />
          <span>Scan Fruit</span>
        </Link>
      </div>
    </nav>
  );
};
export default Navbar;
""")

# 5. styles/navbar.css
write_file(os.path.join(frontend_src, "styles", "navbar.css"), """.navbar {
  background-color: var(--card-bg);
  box-shadow: var(--shadow-sm);
  position: sticky;
  top: 0;
  z-index: 50;
}

.navbar-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 4rem;
}

.navbar-logo {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 700;
  font-size: 1.25rem;
  color: var(--success-dark);
}

.logo-icon {
  color: var(--success);
}

.navbar-links {
  display: flex;
  gap: 2rem;
}

.nav-link {
  color: var(--text-muted);
  font-weight: 500;
  position: relative;
}

.nav-link:hover, .nav-link.active {
  color: var(--success-dark);
}

.nav-link.active::after {
  content: '';
  position: absolute;
  bottom: -1.25rem;
  left: 0;
  width: 100%;
  height: 2px;
  background-color: var(--success);
}

.btn-primary {
  background: linear-gradient(135deg, var(--success), var(--success-dark));
  color: white;
  padding: 0.5rem 1.25rem;
  border-radius: var(--radius-lg);
  font-weight: 500;
  box-shadow: var(--shadow);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: var(--shadow-lg);
}

@media (max-width: 768px) {
  .navbar-links {
    display: none;
  }
}
""")

# 6. App.jsx
write_file(os.path.join(frontend_src, "App.jsx"), """import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Scan from './pages/Scan';
import Dashboard from './pages/Dashboard';
import CompareModels from './pages/CompareModels';
import HistoryPage from './pages/History';
import Research from './pages/Research';
import About from './pages/About';
import './styles/global.css';

function App() {
  return (
    <BrowserRouter>
      <div className="app-wrapper">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/scan" element={<Scan />} />
            <Route path="/compare" element={<CompareModels />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/research" element={<Research />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
export default App;
""")

# 7. main.jsx
write_file(os.path.join(frontend_src, "main.jsx"), """import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './styles/global.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
""")

print("Frontend structure partially created successfully!")
