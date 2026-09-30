import React from 'react';
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
          <Link to="/upload" className={`nav-link ${isActive('/upload') ? 'active' : ''}`}>Upload & Predict</Link>

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
