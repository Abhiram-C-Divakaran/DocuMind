import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  ArrowLeft,
  Check,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Clock,
  Copy,
  FileText,
  History,
  Info,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  ShieldAlert,
  Trash2,
  XCircle,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import {
  DocumentReview,
  PreviewMeta,
  ReviewCorrectionPayload,
  ReviewHistoryResponse,
  StructuredExtractionData,
} from '../../types';
import { StatusBadge } from '../Documents/StatusBadge';
import { ReviewStatusBadge } from '../Documents/ReviewStatusBadge';

interface DocumentReviewWorkspaceProps {
  documentId: string;
  onBack: () => void;
  onReviewUpdated?: () => void;
}

interface FormLineItem {
  description: string;
  quantity: string;
  unit_price: string;
  total_price: string;
}

interface ReviewFormState {
  vendor_name: string;
  vendor_address: string;
  document_date: string;
  total_amount: string;
  currency: string;
  tax_amount: string;
  line_items: FormLineItem[];
  reviewer_notes: string;
}

function toFormState(
  data: StructuredExtractionData | null | undefined,
  notes: string | null | undefined
): ReviewFormState {
  const src = data || {};
  const lineItems = (src.line_items || []).map((item) => ({
    description: item.description != null ? String(item.description) : '',
    quantity: item.quantity != null ? String(item.quantity) : '',
    unit_price: item.unit_price != null ? String(item.unit_price) : '',
    total_price: item.total_price != null ? String(item.total_price) : '',
  }));

  return {
    vendor_name: src.vendor_name != null ? String(src.vendor_name) : '',
    vendor_address: src.vendor_address != null ? String(src.vendor_address) : '',
    document_date: src.document_date != null ? String(src.document_date) : '',
    total_amount: src.total_amount != null ? String(src.total_amount) : '',
    currency: src.currency != null ? String(src.currency) : '',
    tax_amount: src.tax_amount != null ? String(src.tax_amount) : '',
    line_items: lineItems,
    reviewer_notes: notes || '',
  };
}

function toApiCorrectionPayload(form: ReviewFormState): ReviewCorrectionPayload {
  const cleanStr = (val: string) => {
    const trimmed = val.trim();
    return trimmed ? trimmed : null;
  };

  return {
    vendor_name: cleanStr(form.vendor_name),
    vendor_address: cleanStr(form.vendor_address),
    document_date: cleanStr(form.document_date),
    total_amount: cleanStr(form.total_amount),
    currency: cleanStr(form.currency)?.toUpperCase() || null,
    tax_amount: cleanStr(form.tax_amount),
    line_items: form.line_items
      .filter(
        (li) =>
          li.description.trim() ||
          li.quantity.trim() ||
          li.unit_price.trim() ||
          li.total_price.trim()
      )
      .map((li) => ({
        description: cleanStr(li.description),
        quantity: cleanStr(li.quantity),
        unit_price: cleanStr(li.unit_price),
        total_price: cleanStr(li.total_price),
      })),
  };
}

function validateFormClient(form: ReviewFormState): Record<string, string> {
  const errors: Record<string, string> = {};

  if (form.document_date.trim()) {
    if (!/^\d{4}-\d{2}-\d{2}$/.test(form.document_date.trim())) {
      errors.document_date = 'Use ISO format YYYY-MM-DD (e.g. 2024-03-15).';
    }
  }

  let totalNum: number | null = null;
  if (form.total_amount.trim()) {
    const parsed = Number(form.total_amount.trim());
    if (!Number.isFinite(parsed) || parsed <= 0) {
      errors.total_amount = 'Total amount must be greater than zero.';
    } else {
      totalNum = parsed;
    }
  }

  let taxNum: number | null = null;
  if (form.tax_amount.trim()) {
    const parsed = Number(form.tax_amount.trim());
    if (!Number.isFinite(parsed) || parsed < 0) {
      errors.tax_amount = 'Tax amount cannot be negative.';
    } else {
      taxNum = parsed;
    }
  }

  if (totalNum !== null && taxNum !== null && taxNum > totalNum) {
    errors.tax_amount = 'Tax amount cannot exceed total amount.';
  }

  if (form.currency.trim()) {
    if (!/^[A-Za-z]{3}$/.test(form.currency.trim())) {
      errors.currency = 'Enter a 3-letter ISO currency code (e.g. USD, MYR, EUR).';
    }
  }

  return errors;
}

export const DocumentReviewWorkspace: React.FC<DocumentReviewWorkspaceProps> = ({
  documentId,
  onBack,
  onReviewUpdated,
}) => {
  const [review, setReview] = useState<DocumentReview | null>(null);
  const [previewMeta, setPreviewMeta] = useState<PreviewMeta | null>(null);
  const [history, setHistory] = useState<ReviewHistoryResponse | null>(null);

  const [form, setForm] = useState<ReviewFormState>(() => toFormState(null, null));
  const [baselineForm, setBaselineForm] = useState<ReviewFormState>(() =>
    toFormState(null, null)
  );

  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [conflictNotice, setConflictNotice] = useState<string | null>(null);
  const [successNotice, setSuccessNotice] = useState<string | null>(null);

  // Preview state
  const [zoomPct, setZoomPct] = useState(100);
  const [currentPage, setCurrentPage] = useState(1);
  const [previewLoading, setPreviewLoading] = useState(true);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [showRawText, setShowRawText] = useState(false);
  const [copiedText, setCopiedText] = useState(false);

  // Unsaved navigation confirmation
  const [confirmBackOpen, setConfirmBackOpen] = useState(false);

  const isDirty = useMemo(
    () => JSON.stringify(form) !== JSON.stringify(baselineForm),
    [form, baselineForm]
  );

  const clientErrors = useMemo(() => validateFormClient(form), [form]);
  const hasClientErrors = Object.keys(clientErrors).length > 0;

  const isLocked =
    review?.review_status === 'APPROVED' || review?.review_status === 'REJECTED';

  const loadAll = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    setConflictNotice(null);
    setActionError(null);
    try {
      const [revRes, metaRes, histRes] = await Promise.all([
        apiClient.fetchDocumentReview(documentId),
        apiClient.fetchPreviewMeta(documentId),
        apiClient.fetchReviewHistory(documentId),
      ]);
      setReview(revRes);
      setPreviewMeta(metaRes);
      setHistory(histRes);

      const initialData = revRes.corrected_data || revRes.original_data || {};
      const nextForm = toFormState(initialData, revRes.reviewer_notes);
      setForm(nextForm);
      setBaselineForm(nextForm);
      setCurrentPage(1);
      setPreviewLoading(true);
      setPreviewError(null);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load document review.';
      setLoadError(msg);
    } finally {
      setLoading(false);
    }
  }, [documentId]);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = '';
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [isDirty]);

  const handleBackClick = () => {
    if (isDirty) {
      setConfirmBackOpen(true);
    } else {
      onBack();
    }
  };

  const applyServerReviewUpdate = async (updated: DocumentReview, message: string) => {
    setReview(updated);
    const synced = toFormState(
      updated.corrected_data || updated.original_data || {},
      updated.reviewer_notes
    );
    setForm(synced);
    setBaselineForm(synced);
    setSuccessNotice(message);
    setActionError(null);
    setConflictNotice(null);
    try {
      const hist = await apiClient.fetchReviewHistory(documentId);
      setHistory(hist);
    } catch {
      // Non-fatal if history refresh fails
    }
    onReviewUpdated?.();
  };

  const handleApiError = (err: unknown, fallback: string) => {
    const typed = err as Error & { status?: number };
    if (typed.status === 409) {
      setConflictNotice(typed.message || 'Conflict detected while updating review.');
    } else {
      setActionError(typed.message || fallback);
    }
  };

  const handleSaveDraft = async () => {
    if (!review || hasClientErrors) return;
    setSubmitting(true);
    setActionError(null);
    setSuccessNotice(null);
    setConflictNotice(null);
    try {
      const updated = await apiClient.saveReviewDraft(documentId, {
        corrected_data: toApiCorrectionPayload(form),
        reviewer_notes: form.reviewer_notes.trim() || null,
        expected_version: review.version,
      });
      await applyServerReviewUpdate(updated, 'Draft saved (In Review). Original predictions preserved.');
    } catch (err: unknown) {
      handleApiError(err, 'Failed to save draft.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleApprove = async () => {
    if (!review || hasClientErrors) return;
    setSubmitting(true);
    setActionError(null);
    setSuccessNotice(null);
    setConflictNotice(null);
    try {
      const updated = await apiClient.approveReview(documentId, {
        corrected_data: toApiCorrectionPayload(form),
        reviewer_notes: form.reviewer_notes.trim() || null,
        expected_version: review.version,
      });
      await applyServerReviewUpdate(
        updated,
        'Review approved! Verified data is now eligible for active-learning feedback export.'
      );
    } catch (err: unknown) {
      handleApiError(err, 'Failed to approve review.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleReject = async () => {
    if (!review) return;
    setSubmitting(true);
    setActionError(null);
    setSuccessNotice(null);
    setConflictNotice(null);
    try {
      const updated = await apiClient.rejectReview(documentId, {
        corrected_data: toApiCorrectionPayload(form),
        reviewer_notes: form.reviewer_notes.trim() || null,
        expected_version: review.version,
      });
      await applyServerReviewUpdate(
        updated,
        'Review rejected. Document is excluded from active-learning training export.'
      );
    } catch (err: unknown) {
      handleApiError(err, 'Failed to reject review.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleReopen = async () => {
    if (!review) return;
    setSubmitting(true);
    setActionError(null);
    setSuccessNotice(null);
    setConflictNotice(null);
    try {
      const updated = await apiClient.reopenReview(documentId, {
        reviewer_notes: form.reviewer_notes.trim() || null,
        expected_version: review.version,
      });
      await applyServerReviewUpdate(updated, 'Review reopened for editing (In Review).');
    } catch (err: unknown) {
      handleApiError(err, 'Failed to reopen review.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleResetToOriginal = () => {
    if (!review || isLocked) return;
    setForm((prev) => ({
      ...toFormState(review.original_data, prev.reviewer_notes),
    }));
    setSuccessNotice('Form reset to original model predictions (not yet saved).');
  };

  const handleCopyRawText = async () => {
    if (!review?.raw_text) return;
    try {
      await navigator.clipboard.writeText(review.raw_text);
      setCopiedText(true);
      setTimeout(() => setCopiedText(false), 1800);
    } catch {
      // Clipboard blocked
    }
  };

  const isFieldModifiedFromOriginal = (field: keyof StructuredExtractionData): boolean => {
    if (!review?.original_data) return false;
    const origVal = review.original_data[field];
    const origStr = origVal != null ? String(origVal).trim() : '';
    const currVal = form[field as keyof ReviewFormState];
    const currStr = typeof currVal === 'string' ? currVal.trim() : '';
    return origStr !== currStr;
  };

  const renderFieldHeader = (
    label: string,
    fieldKey: keyof StructuredExtractionData,
    requiredForDoc?: boolean
  ) => {
    const modified = isFieldModifiedFromOriginal(fieldKey);
    const fieldConf = review?.field_confidences?.[fieldKey];
    const isMissingOrig = review?.missing_critical_fields.includes(fieldKey);

    return (
      <div className="flex items-center justify-between gap-2 mb-1">
        <label className="text-xs font-semibold text-slate-700 flex items-center gap-1.5">
          <span>{label}</span>
          {requiredForDoc && isMissingOrig && (
            <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200">
              Missing in prediction
            </span>
          )}
          {modified && (
            <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200">
              Edited
            </span>
          )}
        </label>

        {fieldConf !== undefined && fieldConf !== null && (
          <span
            title="Heuristic field confidence (uncalibrated)"
            className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-100 text-slate-600"
          >
            {Math.round(fieldConf * 100)}% heur.
          </span>
        )}
      </div>
    );
  };

  const renderOriginalCaption = (fieldKey: keyof StructuredExtractionData) => {
    const val = review?.original_data?.[fieldKey];
    const display = val != null && String(val).trim() !== '' ? String(val) : 'None predicted';
    return (
      <p className="mt-1 text-[11px] text-slate-500">
        Original prediction:{' '}
        <span className="font-mono text-slate-700 break-all">{display}</span>
      </p>
    );
  };

  if (loading) {
    return (
      <div className="bg-white rounded-xl border border-slate-200 p-8 shadow-sm space-y-4 animate-pulse">
        <div className="h-6 w-64 bg-slate-200 rounded" />
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-6 h-96 bg-slate-100 rounded-xl" />
          <div className="lg:col-span-6 h-96 bg-slate-100 rounded-xl" />
        </div>
      </div>
    );
  }

  if (loadError || !review) {
    return (
      <div className="bg-white rounded-xl border border-rose-200 p-8 shadow-sm text-center space-y-3">
        <XCircle className="w-8 h-8 text-rose-500 mx-auto" />
        <h3 className="text-sm font-bold text-slate-900">Unable to load review workspace</h3>
        <p className="text-xs text-rose-700">{loadError || 'Document review not found.'}</p>
        <div className="flex items-center justify-center gap-3 pt-2">
          <button
            type="button"
            onClick={onBack}
            className="px-3.5 py-2 text-xs font-semibold rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50"
          >
            Back to Queue
          </button>
          <button
            type="button"
            onClick={loadAll}
            className="px-3.5 py-2 text-xs font-semibold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const maxPreviewPages = previewMeta
    ? Math.min(previewMeta.total_pages, previewMeta.max_preview_pages)
    : 1;
  const previewSrc = previewMeta?.is_pdf
    ? apiClient.getDocumentPreviewUrl(documentId, currentPage)
    : apiClient.getDocumentPreviewUrl(documentId);

  return (
    <div className="space-y-4">
      {/* Unsaved confirmation modal */}
      {confirmBackOpen && (
        <div className="fixed inset-0 z-50 bg-slate-900/50 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl border border-slate-200 shadow-xl max-w-md w-full p-5 space-y-4">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-sm font-bold text-slate-900">Unsaved Review Changes</h4>
                <p className="text-xs text-slate-600 mt-1">
                  You have unsaved edits in this workspace. Leaving now will discard your local changes.
                </p>
              </div>
            </div>
            <div className="flex items-center justify-end gap-2">
              <button
                type="button"
                onClick={() => setConfirmBackOpen(false)}
                className="px-3 py-1.5 text-xs font-semibold rounded-lg border border-slate-200 text-slate-700 hover:bg-slate-50"
              >
                Stay in Workspace
              </button>
              <button
                type="button"
                onClick={() => {
                  setConfirmBackOpen(false);
                  onBack();
                }}
                className="px-3 py-1.5 text-xs font-semibold rounded-lg bg-rose-600 text-white hover:bg-rose-700"
              >
                Discard &amp; Leave
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Top Navigation & Status Bar */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm px-4 py-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={handleBackClick}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Back to Queue</span>
          </button>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-sm font-bold text-slate-900">
                {previewMeta?.original_filename || documentId}
              </h2>
              <ReviewStatusBadge status={review.review_status} size="sm" />
              <StatusBadge status={review.document_status} size="sm" />
              {isDirty && (
                <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-amber-100 text-amber-800 border border-amber-200">
                  Unsaved changes
                </span>
              )}
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5 font-mono">
              Doc {documentId.slice(0, 8)} &bull; Engine: {review.extraction_method || 'NONE'} &bull;
              Version v{review.version} &bull; Reviewer: {review.reviewer_identifier}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {review.confidence !== null && review.confidence !== undefined && (
            <span
              title="Heuristic confidence only; not calibrated for statistical correctness"
              className="font-mono px-2.5 py-1 rounded-lg bg-slate-100 text-slate-700 border border-slate-200"
            >
              Overall Heuristic Conf:{' '}
              <strong>{(review.confidence * 100).toFixed(1)}%</strong>
            </span>
          )}
          <button
            type="button"
            onClick={loadAll}
            disabled={submitting}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Reload</span>
          </button>
        </div>
      </div>

      {/* Notices */}
      {conflictNotice && (
        <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0" />
            <span>{conflictNotice}</span>
          </div>
          <button
            type="button"
            onClick={loadAll}
            className="px-2.5 py-1 rounded-md bg-amber-600 text-white font-semibold hover:bg-amber-700 shrink-0"
          >
            Reload Latest Version
          </button>
        </div>
      )}

      {actionError && (
        <div className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{actionError}</span>
        </div>
      )}

      {successNotice && (
        <div className="p-3.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
          <span>{successNotice}</span>
        </div>
      )}

      {/* Split-Screen Workspace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* LEFT PANEL: Document Preview + Metadata + OCR Raw Text */}
        <div className="lg:col-span-6 space-y-4">
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100 bg-slate-50/70 flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-xs font-semibold text-slate-800">
                <FileText className="w-4 h-4 text-indigo-600" />
                <span>Document Preview</span>
                {previewMeta?.is_pdf && (
                  <span className="text-[11px] font-normal text-slate-500">
                    ({previewMeta.total_pages} page{previewMeta.total_pages > 1 ? 's' : ''})
                  </span>
                )}
              </div>

              <div className="flex items-center gap-1.5">
                {previewMeta?.is_pdf && maxPreviewPages > 1 && (
                  <div className="flex items-center gap-1 mr-2 bg-white border border-slate-200 rounded-lg px-1.5 py-0.5 text-xs">
                    <button
                      type="button"
                      disabled={currentPage <= 1}
                      onClick={() => {
                        setPreviewLoading(true);
                        setPreviewError(null);
                        setCurrentPage((p) => Math.max(1, p - 1));
                      }}
                      className="p-1 text-slate-600 hover:text-slate-900 disabled:opacity-30"
                      title="Previous Page"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                    </button>
                    <span className="font-mono text-[11px] text-slate-700 px-1">
                      {currentPage}/{maxPreviewPages}
                    </span>
                    <button
                      type="button"
                      disabled={currentPage >= maxPreviewPages}
                      onClick={() => {
                        setPreviewLoading(true);
                        setPreviewError(null);
                        setCurrentPage((p) => Math.min(maxPreviewPages, p + 1));
                      }}
                      className="p-1 text-slate-600 hover:text-slate-900 disabled:opacity-30"
                      title="Next Page"
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                )}

                <div className="flex items-center gap-1 bg-white border border-slate-200 rounded-lg px-1.5 py-0.5 text-xs">
                  <button
                    type="button"
                    onClick={() => setZoomPct((z) => Math.max(50, z - 25))}
                    disabled={zoomPct <= 50}
                    className="p-1 text-slate-600 hover:text-slate-900 disabled:opacity-30"
                    title="Zoom Out"
                  >
                    <ZoomOut className="w-3.5 h-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => setZoomPct(100)}
                    className="font-mono text-[11px] text-slate-700 px-1 hover:text-indigo-600"
                    title="Reset Zoom"
                  >
                    {zoomPct}%
                  </button>
                  <button
                    type="button"
                    onClick={() => setZoomPct((z) => Math.min(200, z + 25))}
                    disabled={zoomPct >= 200}
                    className="p-1 text-slate-600 hover:text-slate-900 disabled:opacity-30"
                    title="Zoom In"
                  >
                    <ZoomIn className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </div>

            {/* Viewport */}
            <div className="bg-slate-100/90 p-4 min-h-[420px] max-h-[650px] overflow-auto flex items-start justify-center relative">
              {previewLoading && !previewError && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-100/70 z-10 text-xs text-slate-500">
                  Rendering controlled preview...
                </div>
              )}

              {previewError ? (
                <div className="my-auto text-center p-6 space-y-2">
                  <AlertTriangle className="w-6 h-6 text-amber-600 mx-auto" />
                  <p className="text-xs font-semibold text-slate-800">{previewError}</p>
                  <button
                    type="button"
                    onClick={() => {
                      setPreviewError(null);
                      setPreviewLoading(true);
                    }}
                    className="text-xs font-semibold text-indigo-600 underline"
                  >
                    Retry Preview
                  </button>
                </div>
              ) : (
                <img
                  src={previewSrc}
                  alt={previewMeta?.original_filename || 'Document preview'}
                  style={{ width: `${zoomPct}%`, maxWidth: 'none' }}
                  onLoad={() => setPreviewLoading(false)}
                  onError={() => {
                    setPreviewLoading(false);
                    setPreviewError('Could not load document preview from storage.');
                  }}
                  className="rounded shadow-md bg-white border border-slate-200 transition-all duration-150"
                />
              )}
            </div>

            {previewMeta && previewMeta.total_pages > previewMeta.max_preview_pages && (
              <div className="px-4 py-2 bg-amber-50 border-t border-amber-200 text-[11px] text-amber-800">
                Preview capped at first {previewMeta.max_preview_pages} of {previewMeta.total_pages}{' '}
                PDF pages for safety.
              </div>
            )}
          </div>

          {/* Collapsible OCR Raw Text */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-4 py-2.5 flex items-center justify-between bg-slate-50/70 border-b border-slate-100">
              <button
                type="button"
                onClick={() => setShowRawText((s) => !s)}
                className="text-xs font-semibold text-slate-800 flex items-center gap-2 hover:text-indigo-600"
              >
                <span>{showRawText ? 'Hide' : 'Show'} Raw OCR Text</span>
                <span className="text-[11px] font-normal text-slate-500">
                  ({review.raw_text ? `${review.raw_text.length} chars` : 'none'})
                </span>
              </button>

              {review.raw_text && (
                <button
                  type="button"
                  onClick={handleCopyRawText}
                  className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-600 hover:text-slate-900"
                >
                  {copiedText ? (
                    <>
                      <Check className="w-3.5 h-3.5 text-emerald-600" />
                      <span>Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3.5 h-3.5" />
                      <span>Copy</span>
                    </>
                  )}
                </button>
              )}
            </div>

            {showRawText && (
              <pre className="p-4 text-[11px] font-mono text-slate-700 whitespace-pre-wrap max-h-60 overflow-y-auto bg-slate-50/40">
                {review.raw_text || 'No raw OCR text recorded for this document.'}
              </pre>
            )}
          </div>
        </div>

        {/* RIGHT PANEL: Editable Form + Validation + Actions + Audit History */}
        <div className="lg:col-span-6 space-y-4">
          {/* Local Attribution & Provenance Banner */}
          <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-[11px] text-slate-600 flex items-start gap-2.5">
            <Info className="w-4 h-4 text-indigo-500 shrink-0 mt-0.5" />
            <div>
              <span className="font-semibold text-slate-800">
                Local Review Mode (anonymous attribution):
              </span>{' '}
              Authentication is disabled in local development; actions are attributed to{' '}
              <code className="bg-slate-200/70 px-1 rounded">{review.reviewer_identifier}</code>.
              Original extraction output remains immutable alongside your verified corrections.
            </div>
          </div>

          {/* Review Reasons & Validation Warnings */}
          {(review.review_reasons.length > 0 || review.validation_errors.length > 0) && (
            <div className="bg-white rounded-xl border border-amber-200/90 p-4 shadow-sm space-y-2.5">
              {review.review_reasons.length > 0 && (
                <div>
                  <h4 className="text-xs font-bold text-amber-900 flex items-center gap-1.5">
                    <Clock className="w-3.5 h-3.5 text-amber-600" />
                    <span>Why This Document Was Flagged</span>
                  </h4>
                  <ul className="mt-1 space-y-0.5 text-xs text-amber-800 list-disc list-inside">
                    {review.review_reasons.map((reason, idx) => (
                      <li key={idx}>{reason}</li>
                    ))}
                  </ul>
                </div>
              )}

              {review.validation_errors.length > 0 && (
                <div className="pt-1 border-t border-amber-100">
                  <h4 className="text-xs font-bold text-rose-900 flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-rose-600" />
                    <span>Extractor Validation Warnings</span>
                  </h4>
                  <ul className="mt-1 space-y-0.5 text-xs text-rose-800 list-disc list-inside">
                    {review.validation_errors.map((errMsg, idx) => (
                      <li key={idx}>{errMsg}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Editable Correction Form */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-900">
                  Structured Receipt Corrections
                </h3>
                <p className="text-[11px] text-slate-500">
                  {isLocked
                    ? `This review is ${review.review_status.toLowerCase()}. Reopen to make further edits.`
                    : 'Verify and edit extracted fields below before approving for feedback export.'}
                </p>
              </div>
              {!isLocked && (
                <button
                  type="button"
                  onClick={handleResetToOriginal}
                  className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-600 hover:text-indigo-600"
                >
                  <RotateCcw className="w-3 h-3" />
                  <span>Reset to Original</span>
                </button>
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* Vendor Name */}
              <div className="sm:col-span-2">
                {renderFieldHeader('Vendor / Merchant Name', 'vendor_name', true)}
                <input
                  type="text"
                  disabled={isLocked || submitting}
                  value={form.vendor_name}
                  onChange={(e) => setForm({ ...form, vendor_name: e.target.value })}
                  placeholder="e.g. Acme Coffee Roasters"
                  className={`w-full px-3 py-2 text-xs rounded-lg border focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-50 disabled:text-slate-600 ${
                    isFieldModifiedFromOriginal('vendor_name')
                      ? 'border-indigo-300 bg-indigo-50/20'
                      : 'border-slate-200'
                  }`}
                />
                {renderOriginalCaption('vendor_name')}
              </div>

              {/* Vendor Address */}
              <div className="sm:col-span-2">
                {renderFieldHeader('Vendor Address', 'vendor_address')}
                <input
                  type="text"
                  disabled={isLocked || submitting}
                  value={form.vendor_address}
                  onChange={(e) => setForm({ ...form, vendor_address: e.target.value })}
                  placeholder="Street, City, Postal Code"
                  className={`w-full px-3 py-2 text-xs rounded-lg border focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-50 disabled:text-slate-600 ${
                    isFieldModifiedFromOriginal('vendor_address')
                      ? 'border-indigo-300 bg-indigo-50/20'
                      : 'border-slate-200'
                  }`}
                />
                {renderOriginalCaption('vendor_address')}
              </div>

              {/* Document Date */}
              <div>
                {renderFieldHeader('Document Date (YYYY-MM-DD)', 'document_date', true)}
                <input
                  type="text"
                  disabled={isLocked || submitting}
                  value={form.document_date}
                  onChange={(e) => setForm({ ...form, document_date: e.target.value })}
                  placeholder="YYYY-MM-DD"
                  className={`w-full px-3 py-2 text-xs font-mono rounded-lg border focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-50 disabled:text-slate-600 ${
                    clientErrors.document_date
                      ? 'border-rose-400 bg-rose-50/30'
                      : isFieldModifiedFromOriginal('document_date')
                      ? 'border-indigo-300 bg-indigo-50/20'
                      : 'border-slate-200'
                  }`}
                />
                {clientErrors.document_date && (
                  <p className="mt-1 text-[11px] text-rose-600 font-medium">
                    {clientErrors.document_date}
                  </p>
                )}
                {renderOriginalCaption('document_date')}
              </div>

              {/* Currency */}
              <div>
                {renderFieldHeader('Currency (ISO 4217)', 'currency')}
                <input
                  type="text"
                  disabled={isLocked || submitting}
                  value={form.currency}
                  maxLength={3}
                  onChange={(e) =>
                    setForm({ ...form, currency: e.target.value.toUpperCase() })
                  }
                  placeholder="USD / MYR / EUR"
                  className={`w-full px-3 py-2 text-xs font-mono uppercase rounded-lg border focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-50 disabled:text-slate-600 ${
                    clientErrors.currency
                      ? 'border-rose-400 bg-rose-50/30'
                      : isFieldModifiedFromOriginal('currency')
                      ? 'border-indigo-300 bg-indigo-50/20'
                      : 'border-slate-200'
                  }`}
                />
                {clientErrors.currency && (
                  <p className="mt-1 text-[11px] text-rose-600 font-medium">
                    {clientErrors.currency}
                  </p>
                )}
                {renderOriginalCaption('currency')}
              </div>

              {/* Total Amount */}
              <div>
                {renderFieldHeader('Total Amount', 'total_amount', true)}
                <input
                  type="text"
                  disabled={isLocked || submitting}
                  value={form.total_amount}
                  onChange={(e) => setForm({ ...form, total_amount: e.target.value })}
                  placeholder="0.00"
                  className={`w-full px-3 py-2 text-xs font-mono rounded-lg border focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-50 disabled:text-slate-600 ${
                    clientErrors.total_amount
                      ? 'border-rose-400 bg-rose-50/30'
                      : isFieldModifiedFromOriginal('total_amount')
                      ? 'border-indigo-300 bg-indigo-50/20'
                      : 'border-slate-200'
                  }`}
                />
                {clientErrors.total_amount && (
                  <p className="mt-1 text-[11px] text-rose-600 font-medium">
                    {clientErrors.total_amount}
                  </p>
                )}
                {renderOriginalCaption('total_amount')}
              </div>

              {/* Tax Amount */}
              <div>
                {renderFieldHeader('Tax Amount', 'tax_amount')}
                <input
                  type="text"
                  disabled={isLocked || submitting}
                  value={form.tax_amount}
                  onChange={(e) => setForm({ ...form, tax_amount: e.target.value })}
                  placeholder="0.00"
                  className={`w-full px-3 py-2 text-xs font-mono rounded-lg border focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-50 disabled:text-slate-600 ${
                    clientErrors.tax_amount
                      ? 'border-rose-400 bg-rose-50/30'
                      : isFieldModifiedFromOriginal('tax_amount')
                      ? 'border-indigo-300 bg-indigo-50/20'
                      : 'border-slate-200'
                  }`}
                />
                {clientErrors.tax_amount && (
                  <p className="mt-1 text-[11px] text-rose-600 font-medium">
                    {clientErrors.tax_amount}
                  </p>
                )}
                {renderOriginalCaption('tax_amount')}
              </div>
            </div>

            {/* Line Items Section */}
            <div className="pt-2 border-t border-slate-100 space-y-2.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-700">
                  Line Items ({form.line_items.length})
                </span>
                {!isLocked && (
                  <button
                    type="button"
                    onClick={() =>
                      setForm({
                        ...form,
                        line_items: [
                          ...form.line_items,
                          { description: '', quantity: '', unit_price: '', total_price: '' },
                        ],
                      })
                    }
                    className="inline-flex items-center gap-1 text-[11px] font-semibold text-indigo-600 hover:text-indigo-700"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Add Line Item</span>
                  </button>
                )}
              </div>

              {form.line_items.length === 0 ? (
                <p className="text-[11px] text-slate-400 italic">
                  No line items captured.
                </p>
              ) : (
                <div className="space-y-2">
                  {form.line_items.map((item, idx) => (
                    <div
                      key={idx}
                      className="grid grid-cols-12 gap-2 items-center p-2 rounded-lg bg-slate-50 border border-slate-200/80"
                    >
                      <input
                        type="text"
                        disabled={isLocked || submitting}
                        value={item.description}
                        placeholder="Description"
                        onChange={(e) => {
                          const next = [...form.line_items];
                          next[idx] = { ...next[idx], description: e.target.value };
                          setForm({ ...form, line_items: next });
                        }}
                        className="col-span-5 px-2 py-1 text-xs bg-white border border-slate-200 rounded"
                      />
                      <input
                        type="text"
                        disabled={isLocked || submitting}
                        value={item.quantity}
                        placeholder="Qty"
                        onChange={(e) => {
                          const next = [...form.line_items];
                          next[idx] = { ...next[idx], quantity: e.target.value };
                          setForm({ ...form, line_items: next });
                        }}
                        className="col-span-2 px-2 py-1 text-xs font-mono bg-white border border-slate-200 rounded"
                      />
                      <input
                        type="text"
                        disabled={isLocked || submitting}
                        value={item.unit_price}
                        placeholder="Unit"
                        onChange={(e) => {
                          const next = [...form.line_items];
                          next[idx] = { ...next[idx], unit_price: e.target.value };
                          setForm({ ...form, line_items: next });
                        }}
                        className="col-span-2 px-2 py-1 text-xs font-mono bg-white border border-slate-200 rounded"
                      />
                      <input
                        type="text"
                        disabled={isLocked || submitting}
                        value={item.total_price}
                        placeholder="Total"
                        onChange={(e) => {
                          const next = [...form.line_items];
                          next[idx] = { ...next[idx], total_price: e.target.value };
                          setForm({ ...form, line_items: next });
                        }}
                        className="col-span-2 px-2 py-1 text-xs font-mono bg-white border border-slate-200 rounded"
                      />
                      {!isLocked && (
                        <button
                          type="button"
                          onClick={() => {
                            const next = form.line_items.filter((_, i) => i !== idx);
                            setForm({ ...form, line_items: next });
                          }}
                          className="col-span-1 text-slate-400 hover:text-rose-600 flex justify-center"
                          title="Remove item"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Reviewer Notes */}
            <div className="pt-2 border-t border-slate-100">
              <label className="block text-xs font-semibold text-slate-700 mb-1">
                Reviewer Notes
              </label>
              <textarea
                rows={2}
                disabled={isLocked || submitting}
                value={form.reviewer_notes}
                onChange={(e) => setForm({ ...form, reviewer_notes: e.target.value })}
                placeholder="Optional rationale, illegible stamp note, or rejection reason..."
                className="w-full px-3 py-2 text-xs rounded-lg border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500/20 disabled:bg-slate-50"
              />
            </div>

            {/* Action Bar */}
            <div className="pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2.5">
              {isLocked ? (
                <div className="flex items-center justify-between w-full gap-3">
                  <span className="text-xs text-slate-600">
                    Review closed as <strong>{review.review_status}</strong>
                    {review.approved_at ? ` (${new Date(review.approved_at).toLocaleString()})` : ''}.
                  </span>
                  <button
                    type="button"
                    disabled={submitting}
                    onClick={handleReopen}
                    className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50"
                  >
                    <RotateCcw className="w-3.5 h-3.5" />
                    <span>Reopen Review</span>
                  </button>
                </div>
              ) : (
                <>
                  <button
                    type="button"
                    disabled={submitting || hasClientErrors}
                    onClick={handleSaveDraft}
                    className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:opacity-50"
                  >
                    <Save className="w-3.5 h-3.5" />
                    <span>Save Draft</span>
                  </button>

                  <div className="flex items-center gap-2 ml-auto">
                    <button
                      type="button"
                      disabled={submitting}
                      onClick={handleReject}
                      className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg border border-rose-200 bg-rose-50 text-rose-700 hover:bg-rose-100 disabled:opacity-50"
                    >
                      <XCircle className="w-3.5 h-3.5" />
                      <span>Reject</span>
                    </button>
                    <button
                      type="button"
                      disabled={submitting || hasClientErrors}
                      onClick={handleApprove}
                      className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-semibold rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 shadow-xs disabled:opacity-50"
                    >
                      <CheckCircle2 className="w-3.5 h-3.5" />
                      <span>Approve &amp; Verify</span>
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Audit & Field Correction History */}
          <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 space-y-4">
            <div className="flex items-center gap-2 border-b border-slate-100 pb-2.5">
              <History className="w-4 h-4 text-indigo-600" />
              <h3 className="text-sm font-bold text-slate-900">
                Correction Diffs &amp; Audit History
              </h3>
            </div>

            {history && history.field_corrections.length > 0 && (
              <div>
                <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
                  Logged Field Corrections ({history.field_corrections.length})
                </h4>
                <div className="space-y-1.5 max-h-44 overflow-y-auto">
                  {history.field_corrections.map((fc) => (
                    <div
                      key={fc.id}
                      className="text-xs p-2 rounded-lg bg-slate-50 border border-slate-200/70 flex flex-wrap items-center justify-between gap-2"
                    >
                      <div>
                        <span className="font-semibold text-slate-800">{fc.field_name}:</span>{' '}
                        <span className="line-through text-rose-600 font-mono">
                          {fc.original_value || 'null'}
                        </span>{' '}
                        <span className="text-slate-400">&rarr;</span>{' '}
                        <span className="text-emerald-700 font-mono font-semibold">
                          {fc.corrected_value || 'null'}
                        </span>
                      </div>
                      <span className="text-[10px] text-slate-400">
                        {new Date(fc.created_at).toLocaleString()}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div>
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 mb-2">
                Audit Timeline ({history?.audit_events.length || 0})
              </h4>
              {!history || history.audit_events.length === 0 ? (
                <p className="text-xs text-slate-400 italic">No review events logged yet.</p>
              ) : (
                <div className="space-y-2 max-h-56 overflow-y-auto">
                  {history.audit_events.map((ev) => (
                    <div
                      key={ev.id}
                      className="text-xs p-2.5 rounded-lg border border-slate-100 bg-slate-50/50 space-y-1"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-200/80 text-slate-800">
                            {ev.action}
                          </span>
                          {ev.from_status && ev.to_status && (
                            <span className="text-[11px] text-slate-600">
                              {ev.from_status} &rarr; <strong>{ev.to_status}</strong>
                            </span>
                          )}
                        </div>
                        <span className="text-[10px] text-slate-400">
                          {new Date(ev.event_timestamp).toLocaleString()}
                        </span>
                      </div>
                      <div className="text-[11px] text-slate-500 flex flex-wrap items-center gap-2">
                        <span>Actor: {ev.actor_identifier}</span>
                        {ev.changed_fields.length > 0 && (
                          <span>&bull; Fields: {ev.changed_fields.join(', ')}</span>
                        )}
                      </div>
                      {ev.notes && (
                        <p className="text-[11px] text-slate-700 italic">&ldquo;{ev.notes}&rdquo;</p>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

