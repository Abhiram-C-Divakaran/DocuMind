import React from 'react';
import { ReviewStatus } from '../../types';
import {
  CheckCircle2,
  ClipboardCheck,
  Clock,
  FileEdit,
  MinusCircle,
  XCircle,
} from 'lucide-react';

interface ReviewStatusBadgeProps {
  status: ReviewStatus | string;
  size?: 'sm' | 'md';
}

export const ReviewStatusBadge: React.FC<ReviewStatusBadgeProps> = ({
  status,
  size = 'md',
}) => {
  const configs: Record<
    string,
    { label: string; icon: React.ReactNode; classes: string }
  > = {
    NOT_REQUIRED: {
      label: 'Not Required',
      icon: <MinusCircle className="w-3.5 h-3.5 text-slate-500" />,
      classes: 'bg-slate-50 text-slate-600 border-slate-200',
    },
    PENDING_REVIEW: {
      label: 'Needs Review',
      icon: <Clock className="w-3.5 h-3.5 text-amber-600" />,
      classes: 'bg-amber-50 text-amber-700 border-amber-200/80',
    },
    IN_REVIEW: {
      label: 'In Review',
      icon: <FileEdit className="w-3.5 h-3.5 text-indigo-600" />,
      classes: 'bg-indigo-50 text-indigo-700 border-indigo-200/80',
    },
    APPROVED: {
      label: 'Approved',
      icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />,
      classes: 'bg-emerald-50 text-emerald-700 border-emerald-200/80',
    },
    REJECTED: {
      label: 'Rejected',
      icon: <XCircle className="w-3.5 h-3.5 text-rose-600" />,
      classes: 'bg-rose-50 text-rose-700 border-rose-200/80',
    },
  };

  const config = configs[status] || {
    label: status,
    icon: <ClipboardCheck className="w-3.5 h-3.5 text-slate-500" />,
    classes: 'bg-slate-50 text-slate-700 border-slate-200',
  };

  const sizeClasses =
    size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs font-medium';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border shadow-2xs ${sizeClasses} ${config.classes}`}
    >
      {config.icon}
      <span>{config.label}</span>
    </span>
  );
};

