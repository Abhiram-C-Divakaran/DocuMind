import React from 'react';
import { RefreshCw, UploadCloud, LogOut, UserCheck } from 'lucide-react';
import { NavTab } from './Sidebar';
import { User } from '../../types';

interface HeaderProps {
  currentTab: NavTab;
  onOpenUpload: () => void;
  onRefresh: () => void;
  isRefreshing: boolean;
  currentUser: User;
  onLogout: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onOpenUpload,
  onRefresh,
  isRefreshing,
  currentUser,
  onLogout,
}) => {
  const titles: Record<NavTab, { title: string; subtitle: string }> = {
    dashboard: {
      title: 'Platform Overview',
      subtitle: 'Real-time asynchronous ingestion metrics and system health',
    },
    documents: {
      title: 'Document Ledger',
      subtitle: 'Track queued, processing, completed, and failed extraction jobs',
    },
    reviews: {
      title: 'Human Review & Active Learning',
      subtitle: 'Inspect low-confidence extractions, edit predictions, and approve training feedback',
    },
    upload: {
      title: 'Ingestion Portal',
      subtitle: 'Queue receipts and invoices asynchronously with Celery workers and validation guards',
    },
    'ml-architecture': {
      title: 'ML Architecture & Roadmap',
      subtitle: 'SROIE benchmarks, LayoutLMv3 readiness, and Tesseract baseline models',
    },
    users: {
      title: 'Admin, Accounts & Queue Ops',
      subtitle: 'Manage RBAC accounts, trigger outbox recovery, and export approved feedback',
    },
  };

  const { title, subtitle } = titles[currentTab] || titles.dashboard;
  const canUpload = currentUser.role === 'ADMIN' || currentUser.role === 'REVIEWER';

  return (
    <header className="h-16 px-8 bg-white border-b border-slate-200/80 flex items-center justify-between shrink-0">
      <div>
        <h2 className="text-lg font-bold text-slate-900 tracking-tight">{title}</h2>
        <p className="text-xs text-slate-500">{subtitle}</p>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-50 border border-slate-200 text-xs">
          <UserCheck className="w-3.5 h-3.5 text-indigo-600" />
          <span className="font-medium text-slate-700 max-w-[160px] truncate">{currentUser.email}</span>
          <span className="px-1.5 py-0.5 text-[10px] font-bold rounded bg-indigo-100 text-indigo-700">
            {currentUser.role}
          </span>
        </div>

        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          title="Refresh Data"
          className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors border border-slate-200 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>

        {canUpload && currentTab !== 'upload' && (
          <button
            onClick={onOpenUpload}
            className="flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg shadow-sm shadow-indigo-600/20 transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Upload Document</span>
          </button>
        )}

        <button
          onClick={onLogout}
          title="Sign Out"
          className="flex items-center gap-1.5 px-3 py-2 text-xs font-semibold text-slate-600 hover:text-rose-700 hover:bg-rose-50 rounded-lg border border-slate-200 transition-colors"
        >
          <LogOut className="w-3.5 h-3.5" />
          <span className="hidden md:inline">Sign Out</span>
        </button>
      </div>
    </header>
  );
};
