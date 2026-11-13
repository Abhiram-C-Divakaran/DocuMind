import React from 'react';
import { RefreshCw, UploadCloud } from 'lucide-react';
import { NavTab } from './Sidebar';

interface HeaderProps {
  currentTab: NavTab;
  onOpenUpload: () => void;
  onRefresh: () => void;
  isRefreshing: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onOpenUpload,
  onRefresh,
  isRefreshing,
}) => {
  const titles: Record<NavTab, { title: string; subtitle: string }> = {
    dashboard: {
      title: 'Platform Overview',
      subtitle: 'Real-time document intelligence status and telemetry',
    },
    documents: {
      title: 'Document Ledger',
      subtitle: 'Audit trail of uploaded invoices, receipts and structured extractions',
    },
    reviews: {
      title: 'Human Review & Active Learning',
      subtitle: 'Inspect low-confidence extractions, edit predictions, and approve training feedback',
    },
    upload: {
      title: 'Ingestion Portal',
      subtitle: 'Upload receipts and invoices with cryptographic signature validation',
    },
    'ml-architecture': {
      title: 'ML Architecture & Roadmap',
      subtitle: 'SROIE benchmarks, LayoutLMv3 contracts, and Tesseract baseline models',
    },
  };

  const { title, subtitle } = titles[currentTab] || titles.dashboard;

  return (
    <header className="h-16 px-8 bg-white border-b border-slate-200/80 flex items-center justify-between shrink-0">
      <div>
        <h2 className="text-lg font-bold text-slate-900 tracking-tight">{title}</h2>
        <p className="text-xs text-slate-500">{subtitle}</p>
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          title="Refresh Data"
          className="p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition-colors border border-slate-200 disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
        </button>

        {currentTab !== 'upload' && (
          <button
            onClick={onOpenUpload}
            className="flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg shadow-sm shadow-indigo-600/20 transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
          >
            <UploadCloud className="w-4 h-4" />
            <span>Upload Document</span>
          </button>
        )}
      </div>
    </header>
  );
};

