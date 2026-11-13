import React from 'react';
import { DashboardMetrics, ReviewMetrics } from '../../types';
import {
  FileText,
  CheckCircle2,
  Clock,
  AlertOctagon,
  ClipboardCheck,
  DatabaseZap,
  Edit3,
  XCircle,
} from 'lucide-react';
import { Skeleton } from '../Common/Skeleton';

interface MetricsOverviewProps {
  metrics: DashboardMetrics | null;
  reviewMetrics?: ReviewMetrics | null;
  loading: boolean;
  onOpenReviewQueue?: () => void;
}

export const MetricsOverview: React.FC<MetricsOverviewProps> = ({
  metrics,
  reviewMetrics,
  loading,
  onOpenReviewQueue,
}) => {
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
    <div className="space-y-4">
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

      {/* Human-in-the-Loop Review Analytics */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-2xs">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2">
            <ClipboardCheck className="w-4 h-4 text-indigo-600" />
            <h4 className="text-xs font-bold text-slate-800 uppercase tracking-wider">
              Human Review &amp; Active Learning Telemetry
            </h4>
          </div>
          {onOpenReviewQueue && (
            <button
              type="button"
              onClick={onOpenReviewQueue}
              className="text-xs font-semibold text-indigo-600 hover:text-indigo-800"
            >
              Open Review Queue &rarr;
            </button>
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          <div className="p-3 rounded-lg bg-amber-50/70 border border-amber-200/70">
            <div className="flex items-center justify-between text-[11px] font-semibold text-amber-800">
              <span>Awaiting Review</span>
              <Clock className="w-3.5 h-3.5 text-amber-600" />
            </div>
            <div className="mt-1 text-lg font-bold text-slate-900 font-mono">
              {loading ? '...' : reviewMetrics?.awaiting_review ?? 0}
            </div>
            <p className="text-[10px] text-amber-700 mt-0.5">
              {reviewMetrics
                ? `${reviewMetrics.pending_review} pending, ${reviewMetrics.in_review} in review`
                : 'Queue waiting'}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-emerald-50/70 border border-emerald-200/70">
            <div className="flex items-center justify-between text-[11px] font-semibold text-emerald-800">
              <span>Approved</span>
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            </div>
            <div className="mt-1 text-lg font-bold text-slate-900 font-mono">
              {loading ? '...' : reviewMetrics?.approved ?? 0}
            </div>
            <p className="text-[10px] text-emerald-700 mt-0.5">Human-verified records</p>
          </div>

          <div className="p-3 rounded-lg bg-rose-50/70 border border-rose-200/70">
            <div className="flex items-center justify-between text-[11px] font-semibold text-rose-800">
              <span>Rejected</span>
              <XCircle className="w-3.5 h-3.5 text-rose-600" />
            </div>
            <div className="mt-1 text-lg font-bold text-slate-900 font-mono">
              {loading ? '...' : reviewMetrics?.rejected ?? 0}
            </div>
            <p className="text-[10px] text-rose-700 mt-0.5">Excluded from feedback</p>
          </div>

          <div className="p-3 rounded-lg bg-slate-50 border border-slate-200/80">
            <div className="flex items-center justify-between text-[11px] font-semibold text-slate-700">
              <span>Avg Fields Edited</span>
              <Edit3 className="w-3.5 h-3.5 text-indigo-600" />
            </div>
            <div className="mt-1 text-lg font-bold text-slate-900 font-mono">
              {loading ? '...' : (reviewMetrics?.avg_corrected_fields ?? 0).toFixed(2)}
            </div>
            <p className="text-[10px] text-slate-500 mt-0.5">
              {reviewMetrics
                ? `${Math.round(reviewMetrics.review_completion_rate * 100)}% review completion`
                : 'Across reviewed docs'}
            </p>
          </div>

          <div className="p-3 rounded-lg bg-indigo-50/70 border border-indigo-200/70">
            <div className="flex items-center justify-between text-[11px] font-semibold text-indigo-900">
              <span>Feedback Ready</span>
              <DatabaseZap className="w-3.5 h-3.5 text-indigo-600" />
            </div>
            <div className="mt-1 text-lg font-bold text-slate-900 font-mono">
              {loading ? '...' : reviewMetrics?.eligible_for_feedback_export ?? 0}
            </div>
            <p className="text-[10px] text-indigo-700 mt-0.5">
              Approved records for JSONL
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
