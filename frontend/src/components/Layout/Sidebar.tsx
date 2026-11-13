import React from 'react';
import {
  FileText,
  LayoutDashboard,
  UploadCloud,
  Cpu,
  Layers,
  Database,
  CheckCircle2,
  AlertCircle,
  ClipboardCheck,
} from 'lucide-react';
import { HealthStatus } from '../../types';

export type NavTab = 'dashboard' | 'documents' | 'reviews' | 'upload' | 'ml-architecture';

interface SidebarProps {
  currentTab: NavTab;
  onTabChange: (tab: NavTab) => void;
  health: HealthStatus | null;
  awaitingReviewCount?: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentTab,
  onTabChange,
  health,
  awaitingReviewCount = 0,
}) => {
  const navItems = [
    { id: 'dashboard' as NavTab, label: 'Dashboard', icon: LayoutDashboard },
    { id: 'documents' as NavTab, label: 'Document Ledger', icon: FileText },
    {
      id: 'reviews' as NavTab,
      label: 'Review Queue',
      icon: ClipboardCheck,
      badge: awaitingReviewCount > 0 ? awaitingReviewCount : undefined,
    },
    { id: 'upload' as NavTab, label: 'Upload Documents', icon: UploadCloud },
    { id: 'ml-architecture' as NavTab, label: 'ML & Architecture', icon: Cpu },
  ];

  const isHealthy = health?.status === 'healthy';

  return (
    <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 text-slate-300">
      {/* Brand Header */}
      <div className="h-16 px-6 flex items-center gap-3 border-b border-slate-800/80">
        <div className="w-9 h-9 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-md shadow-indigo-500/20">
          <Layers className="w-5 h-5" />
        </div>
        <div>
          <h1 className="text-sm font-semibold tracking-wide text-white">DocuMind</h1>
          <p className="text-[11px] text-slate-400 font-medium">Document Intelligence</p>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex-1 px-3 py-4 space-y-1">
        <div className="px-3 pb-2 text-[11px] font-semibold tracking-wider text-slate-500 uppercase">
          Workspace
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`w-full flex items-center justify-between gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-indigo-600 text-white shadow-sm shadow-indigo-600/30'
                  : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/60'
              }`}
            >
              <span className="flex items-center gap-3">
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.label}</span>
              </span>
              {item.badge !== undefined && (
                <span
                  className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded-full ${
                    isActive
                      ? 'bg-white/20 text-white'
                      : 'bg-amber-500/20 text-amber-300 border border-amber-400/30'
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* System Status Footer */}
      <div className="p-4 border-t border-slate-800/80 bg-slate-950/40">
        <div className="flex items-center justify-between text-xs mb-2">
          <span className="text-slate-400 font-medium flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-slate-500" />
            Backend Service
          </span>
          <span className="flex items-center gap-1 text-[11px]">
            {isHealthy ? (
              <>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400 font-medium">Connected</span>
              </>
            ) : (
              <>
                <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-amber-400 font-medium">Connecting...</span>
              </>
            )}
          </span>
        </div>

        <div className="space-y-1 text-[11px] text-slate-400 font-mono bg-slate-900/90 p-2.5 rounded-md border border-slate-800">
          <div className="flex justify-between">
            <span>DB:</span>
            <span className="text-slate-300 font-sans truncate max-w-[120px]">
              {health?.components?.database || 'Standby'}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Phase:</span>
            <span className="text-indigo-400 font-semibold">3A (HITL Review)</span>
          </div>
        </div>
      </div>
    </aside>
  );
};

