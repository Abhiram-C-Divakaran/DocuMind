import React, { useState } from 'react';
import { DocumentItem } from '../../types';
import { StatusBadge } from './StatusBadge';
import {
  X,
  FileText,
  Calendar,
  DollarSign,
  MapPin,
  Building2,
  AlertTriangle,
  Code2,
  CheckCircle2,
  Layers,
  Cpu,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Percent,
  XCircle,
  Clock,
  Sparkles,
} from 'lucide-react';

interface DocumentDetailModalProps {
  document: DocumentItem | null;
  onClose: () => void;
}

export const DocumentDetailModal: React.FC<DocumentDetailModalProps> = ({
  document,
  onClose,
}) => {
  const [activeTab, setActiveTab] = useState<'structured' | 'validation' | 'raw'>('structured');
  const [ocrTextOpen, setOcrTextOpen] = useState<boolean>(false);
  const [copiedOcr, setCopiedOcr] = useState<boolean>(false);

  if (!document) return null;

  const latestExtraction = document.extractions?.[0];
  const structuredData = latestExtraction?.structured_data;
  const validationErrors = latestExtraction?.validation_errors || [];
  const fieldConfidences = latestExtraction?.field_confidences || {};
  const rawOcrText = latestExtraction?.raw_text || '';
  const isFailed = document.status === 'FAILED';
  const isPending = document.status === 'PENDING';
  const isPlaceholder = latestExtraction?.extraction_method === 'PLACEHOLDER';

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const handleCopyOcr = () => {
    if (!rawOcrText) return;
    navigator.clipboard.writeText(rawOcrText);
    setCopiedOcr(true);
    setTimeout(() => setCopiedOcr(false), 2000);
  };

  const renderConfidenceBadge = (fieldName: string) => {
    const conf = fieldConfidences[fieldName];
    if (conf === undefined || conf === null) return null;

    const percent = Math.round(conf * 100);
    let colorClass = 'bg-emerald-50 text-emerald-700 border-emerald-200';
    if (percent < 50) colorClass = 'bg-rose-50 text-rose-700 border-rose-200';
    else if (percent < 75) colorClass = 'bg-amber-50 text-amber-700 border-amber-200';

    return (
      <span
        title={`Estimated heuristic confidence: ${percent}%`}
        className={`inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded border ${colorClass}`}
      >
        <Percent className="w-2.5 h-2.5" />
        {percent}%
      </span>
    );
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-slate-900/60 backdrop-blur-xs flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl border border-slate-200 max-w-3xl w-full overflow-hidden flex flex-col max-h-[90vh] animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="p-6 border-b border-slate-100 flex items-start justify-between bg-slate-50/50">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-indigo-600 text-white shadow-md shadow-indigo-600/20 shrink-0">
              <FileText className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2.5 flex-wrap">
                <h3 className="text-base font-bold text-slate-900">
                  {document.original_filename}
                </h3>
                <StatusBadge status={document.status} size="sm" />
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-500 font-mono mt-1">
                <span>ID: {document.id}</span>
                <span>•</span>
                <span>{document.content_type}</span>
                <span>•</span>
                <span>{formatFileSize(document.file_size_bytes)}</span>
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 p-1.5 rounded-lg hover:bg-slate-100 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation Tabs */}
        <div className="flex border-b border-slate-200 px-6 bg-slate-50/30 gap-6 text-xs font-semibold">
          {[
            { id: 'structured', label: 'Structured Extraction', icon: Layers },
            { id: 'validation', label: `Validation & Integrity (${validationErrors.length})`, icon: AlertTriangle },
            { id: 'raw', label: 'Raw Audit JSON', icon: Code2 },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`py-3 flex items-center gap-2 border-b-2 transition-all ${isActive
                    ? 'border-indigo-600 text-indigo-600'
                    : 'border-transparent text-slate-500 hover:text-slate-800'
                  }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Content Body */}
        <div className="p-6 overflow-y-auto flex-1 space-y-6">
          {activeTab === 'structured' && (
            <div className="space-y-6">
              {/* Failure Alert */}
              {isFailed && (
                <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-900 text-xs">
                  <div className="flex items-start gap-3">
                    <XCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold text-sm text-rose-950">OCR Processing Failed</p>
                      <p className="mt-1 leading-relaxed whitespace-pre-wrap font-mono text-[11px] text-rose-800 bg-rose-100/60 p-2.5 rounded-md border border-rose-200">
                        {document.error_message || validationErrors[0] || 'Unknown processing error.'}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Placeholder Notice */}
              {isPlaceholder && !isFailed && (
                <div className="p-4 rounded-xl bg-amber-50/80 border border-amber-200 text-amber-900 text-xs flex items-start gap-3">
                  <Clock className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-bold text-sm">Phase 1 Placeholder Ledger Record</p>
                    <p className="mt-0.5 text-amber-800">
                      This document was uploaded in placeholder mode. No fake predictions were fabricated.
                      To run live OCR, select <strong>Tesseract Baseline</strong>.
                    </p>
                  </div>
                </div>
              )}

              {/* Method & Confidence Telemetry Bar */}
              <div className="p-3.5 rounded-xl bg-slate-100/80 border border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-indigo-600 shrink-0" />
                  <span className="text-slate-700">
                    Engine:{' '}
                    <strong className="text-slate-900 font-mono">
                      {latestExtraction?.extraction_method || 'UNKNOWN'}
                    </strong>
                  </span>
                </div>

                <div className="flex items-center gap-4">
                  <span className="text-[11px] text-slate-500 flex items-center gap-1.5">
                    <span>Overall Confidence:</span>
                    {latestExtraction?.confidence !== null && latestExtraction?.confidence !== undefined ? (
                      <span className="font-mono font-bold text-slate-900 bg-white px-2 py-0.5 rounded border border-slate-200 shadow-2xs">
                        {(latestExtraction.confidence * 100).toFixed(1)}%
                      </span>
                    ) : (
                      <span className="italic text-slate-400">N/A</span>
                    )}
                  </span>
                </div>
              </div>

              {/* Top Extracted Entity Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {/* Vendor Name */}
                <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1.5">
                    <div className="flex items-center gap-1.5">
                      <Building2 className="w-4 h-4 text-indigo-500" />
                      <span>Vendor / Merchant</span>
                    </div>
                    {renderConfidenceBadge('vendor_name')}
                  </div>
                  <p className="text-sm font-bold text-slate-900">
                    {structuredData?.vendor_name ? (
                      <span className="text-slate-900">{structuredData.vendor_name}</span>
                    ) : (
                      <span className="text-slate-400 italic font-normal text-xs">
                        {isFailed ? 'Processing failed' : isPending ? 'Pending extraction' : 'Not detected in OCR'}
                      </span>
                    )}
                  </p>
                </div>

                {/* Total Amount */}
                <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1.5">
                    <div className="flex items-center gap-1.5">
                      <DollarSign className="w-4 h-4 text-emerald-500" />
                      <span>Total Amount</span>
                    </div>
                    {renderConfidenceBadge('total_amount')}
                  </div>
                  <p className="text-sm font-bold text-slate-900">
                    {structuredData?.total_amount ? (
                      <span className="font-mono text-emerald-700 bg-emerald-50/60 px-2 py-0.5 rounded border border-emerald-100">
                        {structuredData.currency || ''} {structuredData.total_amount}
                      </span>
                    ) : (
                      <span className="text-slate-400 italic font-normal text-xs">
                        {isFailed ? 'Processing failed' : isPending ? 'Pending extraction' : 'Not detected in OCR'}
                      </span>
                    )}
                  </p>
                </div>

                {/* Document Date */}
                <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1.5">
                    <div className="flex items-center gap-1.5">
                      <Calendar className="w-4 h-4 text-amber-500" />
                      <span>Document Date</span>
                    </div>
                    {renderConfidenceBadge('document_date')}
                  </div>
                  <p className="text-sm font-bold text-slate-900">
                    {structuredData?.document_date ? (
                      <span className="font-mono">{structuredData.document_date}</span>
                    ) : (
                      <span className="text-slate-400 italic font-normal text-xs">
                        {isFailed ? 'Processing failed' : isPending ? 'Pending extraction' : 'Not detected in OCR'}
                      </span>
                    )}
                  </p>
                </div>

                {/* Tax Amount */}
                <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1.5">
                    <div className="flex items-center gap-1.5">
                      <Percent className="w-4 h-4 text-indigo-500" />
                      <span>Tax / VAT / GST</span>
                    </div>
                    {renderConfidenceBadge('tax_amount')}
                  </div>
                  <p className="text-sm font-bold text-slate-900">
                    {structuredData?.tax_amount ? (
                      <span className="font-mono">
                        {structuredData.currency || ''} {structuredData.tax_amount}
                      </span>
                    ) : (
                      <span className="text-slate-400 italic font-normal text-xs">
                        {isFailed ? 'Processing failed' : isPending ? 'Pending extraction' : 'Not detected / None'}
                      </span>
                    )}
                  </p>
                </div>

                {/* Vendor Address */}
                <div className="sm:col-span-2 p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center justify-between text-slate-500 text-xs font-medium mb-1.5">
                    <div className="flex items-center gap-1.5">
                      <MapPin className="w-4 h-4 text-rose-500" />
                      <span>Vendor Address</span>
                    </div>
                    {renderConfidenceBadge('vendor_address')}
                  </div>
                  <p className="text-xs text-slate-800 leading-relaxed">
                    {structuredData?.vendor_address ? (
                      <span>{structuredData.vendor_address}</span>
                    ) : (
                      <span className="text-slate-400 italic font-normal">
                        {isFailed ? 'Processing failed' : isPending ? 'Pending extraction' : 'Not detected in OCR'}
                      </span>
                    )}
                  </p>
                </div>
              </div>

              {/* Line Items Table */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
                    <span>Itemized Line Items</span>
                    <span className="text-[10px] font-semibold bg-indigo-50 text-indigo-600 px-1.5 py-0.5 rounded border border-indigo-100">
                      Experimental
                    </span>
                  </h4>
                </div>

                {structuredData?.line_items && structuredData.line_items.length > 0 ? (
                  <div className="border border-slate-200 rounded-xl overflow-hidden shadow-2xs">
                    <table className="w-full text-xs text-left">
                      <thead className="bg-slate-50 text-[11px] font-semibold text-slate-500 uppercase border-b border-slate-200">
                        <tr>
                          <th className="py-2.5 px-3">Description</th>
                          <th className="py-2.5 px-3 text-right">Qty</th>
                          <th className="py-2.5 px-3 text-right">Unit Price</th>
                          <th className="py-2.5 px-3 text-right">Total</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {structuredData.line_items.map((item, idx) => (
                          <tr key={idx} className="hover:bg-slate-50/50">
                            <td className="py-2.5 px-3 font-medium text-slate-800">
                              {item.description || 'Item'}
                            </td>
                            <td className="py-2.5 px-3 text-right text-slate-600 font-mono">
                              {item.quantity ?? '-'}
                            </td>
                            <td className="py-2.5 px-3 text-right text-slate-600 font-mono">
                              {item.unit_price ?? '-'}
                            </td>
                            <td className="py-2.5 px-3 text-right font-semibold text-slate-900 font-mono">
                              {item.total_price ?? '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="p-4 rounded-xl border border-dashed border-slate-200 text-center text-xs text-slate-500 bg-slate-50/40">
                    Line-item extraction is in experimental mode. No itemized rows were confidently identified in this document.
                  </div>
                )}
              </div>

              {/* Collapsible Original OCR Text Section */}
              <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-2xs">
                <button
                  type="button"
                  onClick={() => setOcrTextOpen(!ocrTextOpen)}
                  className="w-full px-4 py-3 flex items-center justify-between text-xs font-semibold text-slate-700 bg-slate-50/70 hover:bg-slate-100/70 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-indigo-600" />
                    <span>Raw OCR Extracted Text ({rawOcrText ? `${rawOcrText.length} characters` : 'Empty'})</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {ocrTextOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
                  </div>
                </button>

                {ocrTextOpen && (
                  <div className="p-4 border-t border-slate-100 bg-slate-900 text-slate-200 relative">
                    <div className="flex justify-end mb-2">
                      <button
                        onClick={handleCopyOcr}
                        disabled={!rawOcrText}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition-colors"
                      >
                        {copiedOcr ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                            <span>Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span>Copy Text</span>
                          </>
                        )}
                      </button>
                    </div>
                    <pre className="text-xs font-mono whitespace-pre-wrap overflow-x-auto max-h-72 leading-relaxed text-slate-300">
                      {rawOcrText || '(No OCR text was produced or Tesseract was not executed.)'}
                    </pre>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'validation' && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-indigo-50/80 border border-indigo-200 text-indigo-950 text-xs">
                <p className="font-semibold mb-1 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-indigo-600" />
                  Deterministic Validation Checks
                </p>
                <p className="text-indigo-800 leading-relaxed">
                  Every extraction undergoes strict decimal-safe financial validations, positive amount enforcement,
                  tax sanity checks (tax &le; total), date sanity checks, and ambiguity detection.
                </p>
              </div>

              <div>
                <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                  Validation Log & Diagnostics
                </h4>
                {validationErrors.length > 0 ? (
                  <div className="space-y-2">
                    {validationErrors.map((err, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-amber-50/70 border border-amber-200 text-xs text-amber-900 flex items-start gap-2.5"
                      >
                        <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                        <span className="leading-relaxed font-medium">{err}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 rounded-xl border border-emerald-200 bg-emerald-50/40 text-xs text-emerald-900 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span className="font-medium">All validation checks passed with zero integrity anomalies.</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'raw' && (
            <div className="relative">
              <pre className="p-4 rounded-xl bg-slate-900 text-slate-200 text-xs font-mono overflow-x-auto max-h-96 leading-relaxed">
                {JSON.stringify(document, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-100 flex items-center justify-end bg-slate-50/50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-100 rounded-lg border border-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
