import React, { useCallback, useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  ClipboardCheck,
  FileSearch,
  Filter,
  RefreshCw,
  Search,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import { ReviewQueueItem } from '../../types';
import { StatusBadge } from '../Documents/StatusBadge';
import { ReviewStatusBadge } from '../Documents/ReviewStatusBadge';

interface ReviewQueueProps {
  onOpenReview: (documentId: string) => void;
  refreshSignal?: number;
}

export const ReviewQueue: React.FC<ReviewQueueProps> = ({
  onOpenReview,
  refreshSignal = 0,
}) => {
  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const pageSize = 15;

  const [reviewStatus, setReviewStatus] = useState<string>('NEEDS_REVIEW');
  const [extractionMethod, setExtractionMethod] = useState<string>('ALL');
  const [missingField, setMissingField] = useState<string>('ALL');
  const [sortBy, setSortBy] = useState<string>('confidence_asc');
  const [searchInput, setSearchInput] = useState<string>('');
  const [searchQuery, setSearchQuery] = useState<string>('');

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadQueue = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.fetchReviewQueue({
        page,
        page_size: pageSize,
        review_status: reviewStatus,
        extraction_method: extractionMethod === 'ALL' ? undefined : extractionMethod,
        missing_field: missingField === 'ALL' ? undefined : missingField,
        sort_by: sortBy,
        search: searchQuery || undefined,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load review queue.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [page, reviewStatus, extractionMethod, missingField, sortBy, searchQuery]);

  useEffect(() => {
    loadQueue();
  }, [loadQueue, refreshSignal]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    setSearchQuery(searchInput.trim());
  };

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
        {/* Filter Toolbar */}
        <div className="p-4 border-b border-slate-100 bg-slate-50/60 space-y-3">
          <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-indigo-50 text-indigo-600 border border-indigo-100">
                <ClipboardCheck className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  Human-in-the-Loop Review Queue
                </h3>
                <p className="text-xs text-slate-500">
                  Prioritized documents awaiting inspection, validation, or active-learning sign-off
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <form onSubmit={handleSearchSubmit} className="relative">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  placeholder="Filter by filename..."
                  className="pl-8 pr-3 py-1.5 text-xs bg-white border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 w-52"
                />
              </form>
              <button
                type="button"
                onClick={loadQueue}
                disabled={loading}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                <span>Refresh</span>
              </button>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 text-xs">
            <div className="flex items-center gap-1.5 text-slate-500 font-medium">
              <Filter className="w-3.5 h-3.5" />
              <span>Status:</span>
            </div>
            <select
              aria-label="Review Status Filter"
              value={reviewStatus}
              onChange={(e) => {
                setPage(1);
                setReviewStatus(e.target.value);
              }}
              className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:outline-none focus:border-indigo-500"
            >
              <option value="NEEDS_REVIEW">Awaiting Review (Pending / In Review)</option>
              <option value="ALL">All Documents</option>
              <option value="PENDING_REVIEW">Pending Review</option>
              <option value="IN_REVIEW">In Review (Drafts)</option>
              <option value="APPROVED">Approved</option>
              <option value="REJECTED">Rejected</option>
              <option value="NOT_REQUIRED">Not Required</option>
            </select>

            <select
              aria-label="Extraction Method Filter"
              value={extractionMethod}
              onChange={(e) => {
                setPage(1);
                setExtractionMethod(e.target.value);
              }}
              className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:outline-none focus:border-indigo-500"
            >
              <option value="ALL">All Engines</option>
              <option value="TESSERACT_BASELINE">Tesseract Baseline</option>
              <option value="LAYOUTLMV3">LayoutLMv3</option>
              <option value="PLACEHOLDER">Placeholder</option>
            </select>

            <select
              aria-label="Missing Field Filter"
              value={missingField}
              onChange={(e) => {
                setPage(1);
                setMissingField(e.target.value);
              }}
              className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:outline-none focus:border-indigo-500"
            >
              <option value="ALL">Any Critical Fields</option>
              <option value="vendor_name">Missing Vendor Name</option>
              <option value="document_date">Missing Document Date</option>
              <option value="total_amount">Missing Total Amount</option>
            </select>

            <div className="flex items-center gap-1.5 text-slate-500 font-medium ml-auto">
              <ArrowUpDown className="w-3.5 h-3.5" />
              <select
                aria-label="Sort Queue"
                value={sortBy}
                onChange={(e) => {
                  setPage(1);
                  setSortBy(e.target.value);
                }}
                className="bg-white border border-slate-200 rounded-lg px-2.5 py-1.5 text-xs font-medium text-slate-700 focus:outline-none focus:border-indigo-500"
              >
                <option value="confidence_asc">Lowest Heuristic Confidence First</option>
                <option value="confidence_desc">Highest Heuristic Confidence First</option>
                <option value="created_at_desc">Newest Uploaded First</option>
                <option value="created_at_asc">Oldest Uploaded First</option>
                <option value="updated_at_desc">Recently Reviewed First</option>
              </select>
            </div>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="m-4 p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
            <span>{error}</span>
            <button
              type="button"
              onClick={loadQueue}
              className="font-semibold underline hover:text-rose-950"
            >
              Retry
            </button>
          </div>
        )}

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-100 bg-slate-50/40 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                <th className="py-3 px-4">Document</th>
                <th className="py-3 px-4">Engine & Heuristic Conf.</th>
                <th className="py-3 px-4">Extraction State</th>
                <th className="py-3 px-4">Review State</th>
                <th className="py-3 px-4">Review Flags & Signals</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs">
              {loading ? (
                Array.from({ length: 4 }).map((_, idx) => (
                  <tr key={idx} className="animate-pulse">
                    <td className="py-4 px-4">
                      <div className="h-4 w-44 bg-slate-200 rounded" />
                      <div className="h-3 w-28 bg-slate-100 rounded mt-1.5" />
                    </td>
                    <td className="py-4 px-4">
                      <div className="h-4 w-28 bg-slate-200 rounded" />
                    </td>
                    <td className="py-4 px-4">
                      <div className="h-5 w-20 bg-slate-200 rounded-full" />
                    </td>
                    <td className="py-4 px-4">
                      <div className="h-5 w-24 bg-slate-200 rounded-full" />
                    </td>
                    <td className="py-4 px-4">
                      <div className="h-4 w-40 bg-slate-100 rounded" />
                    </td>
                    <td className="py-4 px-4 text-right">
                      <div className="h-7 w-24 bg-slate-200 rounded ml-auto" />
                    </td>
                  </tr>
                ))
              ) : items.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-14 px-4 text-center">
                    <div className="w-11 h-11 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-3 text-slate-400">
                      <FileSearch className="w-5 h-5" />
                    </div>
                    <p className="text-sm font-semibold text-slate-700">
                      No documents match this review filter
                    </p>
                    <p className="text-xs text-slate-500 mt-1">
                      Try switching status filter to &ldquo;All Documents&rdquo; or ingesting a new receipt.
                    </p>
                  </td>
                </tr>
              ) : (
                items.map((item) => {
                  const confPct =
                    item.confidence !== null && item.confidence !== undefined
                      ? Math.round(item.confidence * 100)
                      : null;

                  return (
                    <tr
                      key={item.document_id}
                      onClick={() => onOpenReview(item.document_id)}
                      className="hover:bg-slate-50/80 transition-colors cursor-pointer"
                    >
                      <td className="py-3.5 px-4">
                        <div className="font-semibold text-slate-900 truncate max-w-[230px]">
                          {item.original_filename}
                        </div>
                        <div className="text-[11px] text-slate-400 font-mono mt-0.5">
                          {item.document_id.slice(0, 8)}... &bull; v{item.version}
                        </div>
                      </td>

                      <td className="py-3.5 px-4">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[11px] bg-slate-100 text-slate-700 px-2 py-0.5 rounded">
                            {item.extraction_method || 'UNKNOWN'}
                          </span>
                          {confPct !== null ? (
                            <span
                              title="Heuristic extraction confidence (not a calibrated probability)"
                              className={`font-mono text-[11px] font-bold px-2 py-0.5 rounded border ${
                                confPct < 60
                                  ? 'bg-rose-50 text-rose-700 border-rose-200'
                                  : confPct < 75
                                  ? 'bg-amber-50 text-amber-700 border-amber-200'
                                  : 'bg-emerald-50 text-emerald-700 border-emerald-200'
                              }`}
                            >
                              {confPct}% heur.
                            </span>
                          ) : (
                            <span className="text-[11px] italic text-slate-400">No score</span>
                          )}
                        </div>
                      </td>

                      <td className="py-3.5 px-4">
                        <StatusBadge status={item.document_status} size="sm" />
                      </td>

                      <td className="py-3.5 px-4">
                        <ReviewStatusBadge status={item.review_status} size="sm" />
                      </td>

                      <td className="py-3.5 px-4 max-w-[280px]">
                        <div className="flex flex-wrap gap-1">
                          {item.missing_critical_fields.map((field) => (
                            <span
                              key={field}
                              className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber-50 text-amber-800 border border-amber-200 text-[10px] font-medium"
                            >
                              missing: {field}
                            </span>
                          ))}
                          {item.validation_errors_count > 0 && (
                            <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-rose-50 text-rose-700 border border-rose-200 text-[10px] font-medium">
                              <AlertTriangle className="w-2.5 h-2.5" />
                              {item.validation_errors_count} warning
                              {item.validation_errors_count > 1 ? 's' : ''}
                            </span>
                          )}
                          {item.changed_fields.length > 0 && (
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 text-[10px] font-medium">
                              {item.changed_fields.length} edited
                            </span>
                          )}
                          {item.missing_critical_fields.length === 0 &&
                            item.validation_errors_count === 0 &&
                            item.changed_fields.length === 0 && (
                              <span className="text-[11px] text-slate-400">
                                {item.review_reasons[0] || 'Ready for verification'}
                              </span>
                            )}
                        </div>
                      </td>

                      <td className="py-3.5 px-4 text-right">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onOpenReview(item.document_id);
                          }}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 shadow-2xs transition-colors"
                        >
                          <ClipboardCheck className="w-3.5 h-3.5" />
                          <span>Open Review</span>
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination */}
        <div className="p-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 bg-slate-50/40">
          <div>
            Showing{' '}
            <span className="font-semibold text-slate-700">
              {items.length > 0 ? (page - 1) * pageSize + 1 : 0}
            </span>{' '}
            to{' '}
            <span className="font-semibold text-slate-700">
              {Math.min(page * pageSize, total)}
            </span>{' '}
            of <span className="font-semibold text-slate-700">{total}</span> queue items
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1 || loading}
              className="p-1.5 rounded-md border border-slate-200 text-slate-600 hover:bg-white disabled:opacity-40"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="px-2 font-medium text-slate-700">
              {page} / {totalPages}
            </span>
            <button
              type="button"
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages || loading}
              className="p-1.5 rounded-md border border-slate-200 text-slate-600 hover:bg-white disabled:opacity-40"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

