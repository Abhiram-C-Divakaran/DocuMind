import React from 'react';
import { DocumentItem } from '../../types';
import { StatusBadge } from './StatusBadge';
import { ReviewStatusBadge } from './ReviewStatusBadge';
import { TableRowSkeleton } from '../Common/Skeleton';
import { EmptyState } from '../Common/EmptyState';
import {
  FileText,
  Search,
  Filter,
  Eye,
  ChevronLeft,
  ChevronRight,
  FileCode,
  Calendar,
  ClipboardCheck,
  RotateCcw,
  Trash2,
  Ban,
} from 'lucide-react';

interface DocumentTableProps {
  documents: DocumentItem[];
  total: number;
  page: number;
  pageSize: number;
  loading: boolean;
  search: string;
  statusFilter: string;
  onSearchChange: (val: string) => void;
  onStatusFilterChange: (val: string) => void;
  onPageChange: (newPage: number) => void;
  onSelectDocument: (doc: DocumentItem) => void;
  onReviewDocument?: (doc: DocumentItem) => void;
  onRetryDocument?: (doc: DocumentItem) => void;
  onCancelDocument?: (doc: DocumentItem) => void;
  onDeleteDocument?: (doc: DocumentItem) => void;
  canMutate?: boolean;
  onUploadClick: () => void;
}

export const DocumentTable: React.FC<DocumentTableProps> = ({
  documents,
  total,
  page,
  pageSize,
  loading,
  search,
  statusFilter,
  onSearchChange,
  onStatusFilterChange,
  onPageChange,
  onSelectDocument,
  onReviewDocument,
  onRetryDocument,
  onCancelDocument,
  onDeleteDocument,
  canMutate = true,
  onUploadClick,
}) => {
  const totalPages = Math.ceil(total / pageSize) || 1;

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const formatDate = (isoString: string): string => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-2xs overflow-hidden">
      {/* Table Toolbar */}
      <div className="p-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-3 bg-slate-50/40">
        <div className="relative flex-1 max-w-sm">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search by filename..."
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-xs rounded-lg border border-slate-200 bg-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-400 shrink-0" />
          <select
            value={statusFilter}
            onChange={(e) => onStatusFilterChange(e.target.value)}
            className="text-xs rounded-lg border border-slate-200 bg-white py-2 px-3 text-slate-700 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">All Statuses</option>
            <option value="COMPLETED">Completed</option>
            <option value="PENDING">Pending</option>
            <option value="PROCESSING">Processing</option>
            <option value="FAILED">Failed</option>
          </select>
        </div>
      </div>

      {/* Table Content */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-200 text-[11px] font-semibold text-slate-500 uppercase tracking-wider bg-slate-50/80">
              <th className="py-3.5 px-4">Document Details</th>
              <th className="py-3.5 px-4">MIME / Signature</th>
              <th className="py-3.5 px-4">Size</th>
              <th className="py-3.5 px-4">Pipeline</th>
              <th className="py-3.5 px-4">Processing</th>
              <th className="py-3.5 px-4">Review</th>
              <th className="py-3.5 px-4">Ingested At</th>
              <th className="py-3.5 px-4 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 text-xs text-slate-700">
            {loading ? (
              <>
                <TableRowSkeleton cols={8} />
                <TableRowSkeleton cols={8} />
                <TableRowSkeleton cols={8} />
                <TableRowSkeleton cols={8} />
              </>
            ) : documents.length === 0 ? (
              <tr>
                <td colSpan={8} className="p-8">
                  <EmptyState
                    icon={FileText}
                    title="No documents found"
                    description={
                      search || statusFilter
                        ? 'Try modifying your filter or search query.'
                        : 'Upload your first receipt or invoice to begin structured data extraction.'
                    }
                    actionText={!search && !statusFilter ? 'Upload Document' : undefined}
                    onAction={onUploadClick}
                  />
                </td>
              </tr>
            ) : (
              documents.map((doc) => {
                const latestExtraction = doc.extractions?.[0];
                return (
                  <tr
                    key={doc.id}
                    className="hover:bg-slate-50/80 transition-colors group cursor-pointer"
                    onClick={() => onSelectDocument(doc)}
                  >
                    <td className="py-3.5 px-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-slate-100 text-slate-600 group-hover:bg-indigo-50 group-hover:text-indigo-600 transition-colors">
                          <FileText className="w-4 h-4" />
                        </div>
                        <div>
                          <p className="font-semibold text-slate-900 group-hover:text-indigo-600 transition-colors truncate max-w-xs">
                            {doc.original_filename}
                          </p>
                          <p className="text-[11px] font-mono text-slate-400 truncate max-w-xs">
                            {doc.id}
                          </p>
                        </div>
                      </div>
                    </td>

                    <td className="py-3.5 px-4">
                      <span className="font-mono text-[11px] bg-slate-100 text-slate-600 px-2 py-0.5 rounded">
                        {doc.content_type}
                      </span>
                    </td>

                    <td className="py-3.5 px-4 font-mono text-slate-600">
                      {formatFileSize(doc.file_size_bytes)}
                    </td>

                    <td className="py-3.5 px-4">
                      <span className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-700 bg-slate-100/80 px-2 py-0.5 rounded-md">
                        <FileCode className="w-3 h-3 text-slate-400" />
                        {latestExtraction?.extraction_method || 'PENDING'}
                      </span>
                    </td>

                    <td className="py-3.5 px-4">
                      <StatusBadge status={doc.status} size="sm" />
                    </td>

                    <td className="py-3.5 px-4">
                      <ReviewStatusBadge status={doc.review_status || 'NOT_REQUIRED'} size="sm" />
                    </td>

                    <td className="py-3.5 px-4 text-slate-500 text-[11px]">
                      <div className="flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5 text-slate-400" />
                        <span>{formatDate(doc.created_at)}</span>
                      </div>
                    </td>

                    <td className="py-3.5 px-4 text-right">
                      <div className="inline-flex items-center gap-1">
                        {canMutate &&
                          onCancelDocument &&
                          (doc.status === 'PENDING' || doc.status === 'PROCESSING') && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onCancelDocument(doc);
                              }}
                              title="Cancel processing job"
                              className="inline-flex items-center gap-1 px-2 py-1.5 rounded-md text-xs font-semibold text-amber-700 hover:bg-amber-50 transition-colors"
                            >
                              <Ban className="w-3.5 h-3.5" />
                            </button>
                          )}
                        {canMutate &&
                          onRetryDocument &&
                          doc.status === 'FAILED' && (
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                onRetryDocument(doc);
                              }}
                              title="Retry failed job"
                              className="inline-flex items-center gap-1 px-2 py-1.5 rounded-md text-xs font-semibold text-sky-700 hover:bg-sky-50 transition-colors"
                            >
                              <RotateCcw className="w-3.5 h-3.5" />
                            </button>
                          )}
                        {canMutate && onReviewDocument && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onReviewDocument(doc);
                            }}
                            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-semibold text-emerald-700 hover:text-emerald-800 hover:bg-emerald-50 transition-colors"
                          >
                            <ClipboardCheck className="w-3.5 h-3.5" />
                            <span>Review</span>
                          </button>
                        )}
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSelectDocument(doc);
                          }}
                          className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-xs font-semibold text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 transition-colors"
                        >
                          <Eye className="w-3.5 h-3.5" />
                          <span>Inspect</span>
                        </button>
                        {canMutate && onDeleteDocument && doc.status !== 'PROCESSING' && (
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteDocument(doc);
                            }}
                            title="Delete document"
                            className="inline-flex items-center gap-1 px-2 py-1.5 rounded-md text-xs font-semibold text-rose-600 hover:bg-rose-50 transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer */}
      <div className="p-4 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500 bg-slate-50/30">
        <div>
          Showing{' '}
          <span className="font-semibold text-slate-700">
            {documents.length > 0 ? (page - 1) * pageSize + 1 : 0}
          </span>{' '}
          to{' '}
          <span className="font-semibold text-slate-700">
            {Math.min(page * pageSize, total)}
          </span>{' '}
          of <span className="font-semibold text-slate-700">{total}</span> documents
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onPageChange(page - 1)}
            disabled={page <= 1 || loading}
            className="p-1.5 rounded-md border border-slate-200 text-slate-600 hover:bg-white hover:text-slate-900 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="px-2 font-medium text-slate-700">
            {page} / {totalPages}
          </span>
          <button
            onClick={() => onPageChange(page + 1)}
            disabled={page >= totalPages || loading}
            className="p-1.5 rounded-md border border-slate-200 text-slate-600 hover:bg-white hover:text-slate-900 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
};
