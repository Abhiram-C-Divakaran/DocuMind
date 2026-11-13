import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  AlertCircle,
  X,
  FileCheck,
  ShieldCheck,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import { DocumentUploadResponse } from '../../types';

interface DocumentUploaderProps {
  onUploadSuccess: (response: DocumentUploadResponse) => void;
  onCancel?: () => void;
}

export const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  onUploadSuccess,
  onCancel,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [extractionMethod, setExtractionMethod] = useState<string>('TESSERACT_BASELINE');
  const [requestReview, setRequestReview] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [validationWarning, setValidationWarning] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const allowedTypes = ['application/pdf', 'image/png', 'image/jpeg'];
  const allowedExts = ['.pdf', '.png', '.jpg', '.jpeg'];
  const maxBytes = 15 * 1024 * 1024; // 15MB

  const handleFileSelection = (file: File) => {
    setUploadError(null);
    setValidationWarning(null);

    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!allowedExts.includes(ext)) {
      setUploadError(`Invalid extension '${ext}'. Only PDF, PNG, and JPEG documents are permitted.`);
      setSelectedFile(null);
      return;
    }

    if (file.size > maxBytes) {
      setUploadError(`File size ${(file.size / (1024 * 1024)).toFixed(2)} MB exceeds maximum limit of 15 MB.`);
      setSelectedFile(null);
      return;
    }

    if (!allowedTypes.includes(file.type) && file.type !== '') {
      setValidationWarning(`Reported browser MIME '${file.type}' will be verified against binary magic signature on server.`);
    }

    setSelectedFile(file);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleUploadSubmit = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setUploadError(null);

    try {
      const res = await apiClient.uploadDocument(selectedFile, extractionMethod, requestReview);
      onUploadSuccess(res);
      setSelectedFile(null);
      setRequestReview(false);
    } catch (err: unknown) {
      const error = err as Error & { details?: Record<string, unknown> };
      setUploadError(error.message || 'Failed to upload document.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 max-w-2xl mx-auto">
      <div className="flex items-center justify-between pb-4 border-b border-slate-100 mb-6">
        <div>
          <h3 className="text-base font-bold text-slate-900">Ingest Scanned Document</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Cryptographically validated against magic byte signatures (PDF, PNG, JPEG)
          </p>
        </div>
        {onCancel && (
          <button
            onClick={onCancel}
            className="text-slate-400 hover:text-slate-600 p-1 rounded-md transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* Drag & Drop Box */}
      {!selectedFile ? (
        <div
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
            dragActive
              ? 'border-indigo-500 bg-indigo-50/50 scale-[0.99]'
              : 'border-slate-300 hover:border-slate-400 bg-slate-50/50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            accept=".pdf,.png,.jpg,.jpeg"
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleFileSelection(e.target.files[0]);
              }
            }}
          />

          <div className="p-3.5 bg-indigo-50 text-indigo-600 rounded-full shadow-2xs mb-3">
            <UploadCloud className="w-8 h-8" />
          </div>
          <p className="text-sm font-semibold text-slate-800">
            Click to upload or drag and drop receipt / invoice
          </p>
          <p className="text-xs text-slate-500 mt-1">
            PDF, PNG or JPEG files up to 15 MB
          </p>
          <div className="mt-4 flex items-center gap-2 text-[11px] text-slate-500 bg-slate-100 px-3 py-1 rounded-full">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            <span>Strict binary signature verification & UUID path isolation</span>
          </div>
        </div>
      ) : (
        /* Selected File Card */
        <div className="p-4 rounded-xl border border-indigo-100 bg-indigo-50/30">
          <div className="flex items-start justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-lg bg-indigo-600 text-white">
                <FileCheck className="w-6 h-6" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-900 truncate max-w-sm">
                  {selectedFile.name}
                </p>
                <div className="flex items-center gap-3 text-xs text-slate-500 mt-0.5">
                  <span>{(selectedFile.size / 1024).toFixed(1)} KB</span>
                  <span>•</span>
                  <span className="font-mono uppercase">{selectedFile.name.split('.').pop()}</span>
                </div>
              </div>
            </div>

            <button
              onClick={() => setSelectedFile(null)}
              disabled={isUploading}
              className="text-slate-400 hover:text-slate-600 p-1 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Validation notice */}
          {validationWarning && (
            <div className="mt-3 p-2.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-xs flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" />
              <span>{validationWarning}</span>
            </div>
          )}

          {/* Extraction Method Selector */}
          <div className="mt-5 pt-4 border-t border-indigo-100/60">
            <label className="block text-xs font-semibold text-slate-700 uppercase tracking-wider mb-2">
              Extraction Pipeline Interface
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              {[
                {
                  id: 'TESSERACT_BASELINE',
                  label: 'Tesseract OCR',
                  badge: 'Phase 2A Active',
                  desc: 'Real OCR + rule-based receipt extraction',
                  disabled: false,
                },
                {
                  id: 'PLACEHOLDER',
                  label: 'Placeholder Only',
                  badge: 'Ledger Only',
                  desc: 'Records upload record without inference',
                  disabled: false,
                },
                {
                  id: 'LAYOUTLMV3',
                  label: 'LayoutLMv3',
                  badge: 'Multimodal ML',
                  desc: 'Fine-tuned transformer (requires checkpoint)',
                  disabled: false,
                },
              ].map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  disabled={isUploading || opt.disabled}
                  onClick={() => !opt.disabled && setExtractionMethod(opt.id)}
                  className={`p-3 text-left rounded-lg border text-xs transition-all ${
                    opt.disabled
                      ? 'border-slate-200 bg-slate-50 opacity-60 cursor-not-allowed'
                      : extractionMethod === opt.id
                      ? 'border-indigo-600 bg-indigo-50/80 shadow-2xs'
                      : 'border-slate-200 hover:border-slate-300 bg-white'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-semibold text-slate-800">{opt.label}</span>
                    <span
                      className={`text-[10px] font-medium px-1.5 py-0.5 rounded ${
                        opt.id === 'TESSERACT_BASELINE'
                          ? 'bg-emerald-100 text-emerald-800'
                          : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      {opt.badge}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 leading-tight">{opt.desc}</p>
                </button>
              ))}
            </div>

            <label className="mt-3 flex items-center gap-2 text-xs text-slate-600 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={requestReview}
                disabled={isUploading}
                onChange={(e) => setRequestReview(e.target.checked)}
                className="rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <span>Flag for manual human review regardless of confidence threshold</span>
            </label>
          </div>
        </div>
      )}

      {/* Error Message */}
      {uploadError && (
        <div className="mt-4 p-3 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span className="font-medium">{uploadError}</span>
        </div>
      )}

      {/* Actions */}
      <div className="mt-6 flex items-center justify-end gap-3 pt-4 border-t border-slate-100">
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            disabled={isUploading}
            className="px-4 py-2 text-xs font-semibold text-slate-600 hover:text-slate-800 rounded-lg transition-colors border border-slate-200"
          >
            Cancel
          </button>
        )}
        <button
          type="button"
          onClick={handleUploadSubmit}
          disabled={!selectedFile || isUploading}
          className="flex items-center gap-2 px-5 py-2.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 rounded-lg shadow-sm shadow-indigo-600/20 transition-all focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2"
        >
          {isUploading ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              <span>Verifying & Ingesting...</span>
            </>
          ) : (
            <>
              <UploadCloud className="w-4 h-4" />
              <span>Submit for Ingestion</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};
