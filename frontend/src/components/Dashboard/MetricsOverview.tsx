import React from 'react';
import { DashboardMetrics } from '../../types';
import { FileText, CheckCircle2, Clock, AlertOctagon } from 'lucide-react';
import { Skeleton } from '../Common/Skeleton';

interface MetricsOverviewProps {
  metrics: DashboardMetrics | null;
  loading: boolean;
}

export const MetricsOverview: React.FC<MetricsOverviewProps> = ({ metrics, loading }) => {
  const cards = [
    {
      title: 'Total Ingested',
      value: metrics?.total_documents ?? 0,
      icon: FileText,
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-50',
      borderColor: 'border-indigo-100',
      description: 'PDF, PNG & JPEG files',
    },
    {
      title: 'Extractions Recorded',
      value: metrics?.completed ?? 0,
      icon: CheckCircle2,
      color: 'text-emerald-600',
      bgColor: 'bg-emerald-50',
      borderColor: 'border-emerald-100',
      description: 'Structured ledger entries',
    },
    {
      title: 'Queued / Processing',
      value: (metrics?.pending ?? 0) + (metrics?.processing ?? 0),
      icon: Clock,
      color: 'text-amber-600',
      bgColor: 'bg-amber-50',
      borderColor: 'border-amber-100',
      description: 'Awaiting pipeline pass',
    },
    {
      title: 'Failed Validation',
      value: metrics?.failed ?? 0,
      icon: AlertOctagon,
      color: 'text-rose-600',
      bgColor: 'bg-rose-50',
      borderColor: 'border-rose-100',
      description: 'MIME or format rejections',
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, idx) => {
        const Icon = card.icon;
        return (
          <div
            key={idx}
            className={`p-5 bg-white rounded-xl border ${card.borderColor} shadow-2xs transition-all hover:shadow-sm`}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                {card.title}
              </span>
              <div className={`p-2 rounded-lg ${card.bgColor} ${card.color}`}>
                <Icon className="w-4 h-4" />
              </div>
            </div>

            <div className="mt-3">
              {loading ? (
                <Skeleton className="h-8 w-16 my-1" />
              ) : (
                <div className="flex items-baseline gap-2">
                  <span className="text-2xl font-bold text-slate-900 tracking-tight">
                    {card.value.toLocaleString()}
                  </span>
                </div>
              )}
              <p className="mt-1 text-xs text-slate-400 font-medium">{card.description}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
};
