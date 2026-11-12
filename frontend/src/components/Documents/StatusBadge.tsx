import React from 'react';
import { DocumentStatus } from '../../types';
import { CheckCircle2, Clock, AlertTriangle, XCircle } from 'lucide-react';

interface StatusBadgeProps {
  status: DocumentStatus;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  const configs: Record<DocumentStatus, { label: string; icon: React.ReactNode; classes: string }> = {
    COMPLETED: {
      label: 'Completed',
      icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />,
      classes: 'bg-emerald-50 text-emerald-700 border-emerald-200/80',
    },
    PENDING: {
      label: 'Pending',
      icon: <Clock className="w-3.5 h-3.5 text-amber-600" />,
      classes: 'bg-amber-50 text-amber-700 border-amber-200/80',
    },
    PROCESSING: {
      label: 'Processing',
      icon: <Clock className="w-3.5 h-3.5 text-indigo-600 animate-spin" />,
      classes: 'bg-indigo-50 text-indigo-700 border-indigo-200/80',
    },
    FAILED: {
      label: 'Failed',
      icon: <XCircle className="w-3.5 h-3.5 text-rose-600" />,
      classes: 'bg-rose-50 text-rose-700 border-rose-200/80',
    },
  };

  const config = configs[status] || {
    label: status,
    icon: <AlertTriangle className="w-3.5 h-3.5 text-slate-500" />,
    classes: 'bg-slate-50 text-slate-700 border-slate-200',
  };

  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs font-medium';

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border shadow-2xs ${sizeClasses} ${config.classes}`}>
      {config.icon}
      <span>{config.label}</span>
    </span>
  );
};

