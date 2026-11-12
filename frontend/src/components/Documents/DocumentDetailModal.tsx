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

  if (!document) return null;

  const latestExtraction = document.extractions?.[0];
  const structuredData = latestExtraction?.structured_data;
  const validationErrors = latestExtraction?.validation_errors || [];

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
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
            { id: 'validation', label: `Validation & Status (${validationErrors.length})`, icon: AlertTriangle },
            { id: 'raw', label: 'Raw Audit JSON', icon: Code2 },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`py-3 flex items-center gap-2 border-b-2 transition-all ${
                  isActive
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
              {/* Method banner */}
              <div className="p-3.5 rounded-xl bg-slate-100/80 border border-slate-200 flex items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-indigo-600 shrink-0" />
                  <span className="text-slate-700">
                    Extraction Pipeline:{' '}
                    <strong className="text-slate-900 font-mono">
                      {latestExtraction?.extraction_method || 'PENDING'}
                    </strong>
                  </span>
                </div>
                <span className="text-[11px] font-mono text-slate-500">
                  Confidence:{' '}
                  {latestExtraction?.confidence !== null && latestExtraction?.confidence !== undefined
                    ? `${(latestExtraction.confidence * 100).toFixed(1)}%`
                    : 'N/A (Phase 1 contract)'}
                </span>
              </div>

              {/* Top Entities Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center gap-2 text-slate-500 text-xs font-medium mb-1">
                    <Building2 className="w-4 h-4 text-indigo-500" />
                    <span>Vendor / Merchant</span>
                  </div>
                  <p className="text-sm font-bold text-slate-900">
                    {structuredData?.vendor_name || (
                      <span className="text-slate-400 italic font-normal">Not extracted (Placeholder mode)</span>
                    )}
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center gap-2 text-slate-500 text-xs font-medium mb-1">
                    <DollarSign className="w-4 h-4 text-emerald-500" />
                    <span>Total Amount</span>
                  </div>
                  <p className="text-sm font-bold text-slate-900">
                    {structuredData?.total_amount ? (
                      `${structuredData.currency || ''} ${structuredData.total_amount}`
                    ) : (
                      <span className="text-slate-400 italic font-normal">Not extracted (Placeholder mode)</span>
                    )}
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center gap-2 text-slate-500 text-xs font-medium mb-1">
                    <Calendar className="w-4 h-4 text-amber-500" />
                    <span>Document Date</span>
                  </div>
                  <p className="text-sm font-bold text-slate-900">
                    {structuredData?.document_date || (
                      <span className="text-slate-400 italic font-normal">Not extracted (Placeholder mode)</span>
                    )}
                  </p>
                </div>

                <div className="p-4 rounded-xl border border-slate-200 bg-white shadow-2xs">
                  <div className="flex items-center gap-2 text-slate-500 text-xs font-medium mb-1">
                    <MapPin className="w-4 h-4 text-rose-500" />
                    <span>Vendor Address</span>
                  </div>
                  <p className="text-sm font-bold text-slate-900 truncate">
                    {structuredData?.vendor_address || (
                      <span className="text-slate-400 italic font-normal">Not extracted (Placeholder mode)</span>
                    )}
                  </p>
                </div>
              </div>

              {/* Line Items Table */}
              <div>
                <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                  Itemized Line Items
                </h4>
                {structuredData?.line_items && structuredData.line_items.length > 0 ? (
                  <div className="border border-slate-200 rounded-xl overflow-hidden">
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
                  <div className="p-4 rounded-xl border border-dashed border-slate-200 text-center text-xs text-slate-500 bg-slate-50/50">
                    No itemized line items recorded. Line items schema is verified for Phase 2 fine-tuning.
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'validation' && (
            <div className="space-y-4">
              <div className="p-4 rounded-xl bg-amber-50/80 border border-amber-200 text-amber-900 text-xs">
                <p className="font-semibold mb-1 flex items-center gap-1.5">
                  <AlertTriangle className="w-4 h-4 text-amber-600" />
                  Validation Rules & Data Integrity Checks
                </p>
                <p className="text-amber-800 leading-relaxed">
                  Phase 1 validates file magic bytes, MIME signatures, and persistence contracts.
                  Extraction models will undergo SROIE dataset evaluation and calibration in Phase 2.
                </p>
              </div>

              <div>
                <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
                  System Diagnostics & Warnings
                </h4>
                {validationErrors.length > 0 ? (
                  <div className="space-y-2">
                    {validationErrors.map((err, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-slate-50 border border-slate-200 text-xs text-slate-700 flex items-start gap-2.5"
                      >
                        <div className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-1.5 shrink-0" />
                        <span className="leading-relaxed">{err}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-4 rounded-xl border border-slate-200 text-xs text-slate-500 flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    <span>No validation anomalies recorded for this document.</span>
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

