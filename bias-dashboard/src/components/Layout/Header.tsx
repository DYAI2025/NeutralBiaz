import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  ShieldCheckIcon,
  HomeIcon,
  DocumentTextIcon,
  ChartBarIcon,
  CogIcon
} from '@heroicons/react/24/outline';
import { useHealthCheck } from '../../hooks/useBiasDetection';

const Header: React.FC = () => {
  const location = useLocation();
  const { data: healthStatus } = useHealthCheck();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Analysis', href: '/analysis', icon: DocumentTextIcon },
    { name: 'History', href: '/history', icon: ChartBarIcon },
    { name: 'Settings', href: '/settings', icon: CogIcon },
  ];

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <ShieldCheckIcon className="h-8 w-8 text-primary-600" />
              <span className="text-xl font-bold text-gray-900">
                BiasNeutralize AI
              </span>
            </Link>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex space-x-8">
            {navigation.map((item) => {
              const Icon = item.icon;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center space-x-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                    isActive(item.href)
                      ? 'bg-primary-100 text-primary-700'
                      : 'text-gray-700 hover:text-primary-600 hover:bg-gray-100'
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  <span>{item.name}</span>
                </Link>
              );
            })}
          </nav>

          {/* Status indicator */}
          <div className="flex items-center space-x-2">
            <div className="flex items-center space-x-1">
              <div
                className={`w-2 h-2 rounded-full ${
                  healthStatus?.status === 'healthy'
                    ? 'bg-green-500'
                    : 'bg-red-500'
                }`}
              />
              <span className="text-xs text-gray-500">
                API {healthStatus?.status === 'healthy' ? 'Online' : 'Offline'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile navigation */}
      <div className="md:hidden">
        <nav className="px-4 pt-2 pb-3 space-y-1 bg-gray-50 border-t border-gray-200">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center space-x-2 px-3 py-2 rounded-md text-sm font-medium ${
                  isActive(item.href)
                    ? 'bg-primary-100 text-primary-700'
                    : 'text-gray-700 hover:text-primary-600 hover:bg-gray-100'
                }`}
              >
                <Icon className="h-4 w-4" />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
};

export default Header;