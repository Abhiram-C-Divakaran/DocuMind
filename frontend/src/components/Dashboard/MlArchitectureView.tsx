import React, { useEffect, useState } from 'react';
import { Cpu, FileText, CheckCircle2, Layers, AlertTriangle } from 'lucide-react';
import { apiClient } from '../../api/client';
import { ModelReadinessResponse } from '../../types';

export const MlArchitectureView: React.FC = () => {
  const [readiness, setReadiness] = useState<ModelReadinessResponse | null>(null);

  useEffect(() => {
    let active = true;
    apiClient
      .fetchModelReadiness()
      .then((res) => {
        if (active) setReadiness(res);
      })
      .catch(() => {
        // Leave readiness null cleanly if backend is offline
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      {/* Intro Banner */}
      <div className="p-6 rounded-2xl bg-indigo-900 text-white shadow-md relative overflow-hidden">
        <div className="relative z-10 max-w-2xl">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-indigo-800 text-indigo-200 text-xs font-semibold mb-3">
            <Cpu className="w-3.5 h-3.5" />
            DocuMind ML Architecture & Model Readiness
          </span>
          <h2 className="text-xl font-bold tracking-tight text-white mb-2">
            Multimodal Intelligence Pipeline
          </h2>
          <p className="text-xs text-indigo-200 leading-relaxed">
            DocuMind combines a deterministic Tesseract OCR + rules default pipeline with a
            LayoutLMv3 multimodal token classifier (`COMPANY`, `DATE`, `ADDRESS`, `TOTAL`), backed by
            human-in-the-loop review and strict checkpoint validation.
          </p>
        </div>
      </div>

      {/* Live Readiness Banner */}
      {readiness && (
        <div
          className={`p-4 rounded-xl border text-xs flex items-start gap-3 ${
            readiness.layoutlmv3_is_production_ready
              ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
              : 'bg-amber-50 border-amber-200 text-amber-900'
          }`}
        >
          {readiness.layoutlmv3_is_production_ready ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
          ) : (
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          )}
          <div>
            <p className="font-semibold">
              LayoutLMv3 Status:{' '}
              {readiness.layoutlmv3_is_production_ready
                ? `Checkpoint Verified (${readiness.checkpoint_identity})`
                : readiness.layoutlmv3_ready
                ? `Synthetic Test Checkpoint Only (${readiness.checkpoint_identity}) — Not Production Ready`
                : 'Not Ready for Inference (Using Tesseract OCR + Rules Default)'}
            </p>
            {readiness.layoutlmv3_reason && (
              <p className="mt-0.5 opacity-90">{readiness.layoutlmv3_reason}</p>
            )}
          </div>
        </div>
      )}

      {/* Comparison Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Baseline Card */}
        <div className="p-6 rounded-xl border border-slate-200 bg-white shadow-2xs">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2.5 rounded-lg bg-amber-50 text-amber-700">
              <FileText className="w-5 h-5" />
            </div>
            <span className="text-[11px] font-semibold bg-amber-100 text-amber-800 px-2 py-0.5 rounded-full">
              Baseline Contract
            </span>
          </div>

          <h3 className="text-base font-bold text-slate-900 mb-1">
            Tesseract OCR + Heuristic Rules
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Classical computer vision & text-regular expression pipeline for tabular receipt recognition.
          </p>

          <div className="space-y-2.5 text-xs text-slate-600 border-t border-slate-100 pt-4">
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <span>Uses Pillow for image pre-processing (binarization, deskewing)</span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <span>Extracts word tokens with spatial bounding boxes (hOCR format)</span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <span>Rule-based anchor matching for vendor, dates, and currency totals</span>
            </div>
          </div>
        </div>

        {/* Transformer Card */}
        <div className="p-6 rounded-xl border border-slate-200 bg-white shadow-2xs">
          <div className="flex items-center justify-between mb-4">
            <div className="p-2.5 rounded-lg bg-indigo-50 text-indigo-700">
              <Layers className="w-5 h-5" />
            </div>
            <span className="text-[11px] font-semibold bg-indigo-100 text-indigo-800 px-2 py-0.5 rounded-full">
              Target Model
            </span>
          </div>

          <h3 className="text-base font-bold text-slate-900 mb-1">
            Multimodal LayoutLMv3
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            Pre-trained multimodal transformer joint modeling visual tokens, text, and 2D spatial layouts.
          </p>

          <div className="space-y-2.5 text-xs text-slate-600 border-t border-slate-100 pt-4">
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <span>Designed for SROIE receipt dataset fine-tuning via Colab GPU / PyTorch</span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <span>BIO entity classification for Company, Date, Address, and Total</span>
            </div>
            <div className="flex items-start gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
              <span>Strict checkpoint schema, label-map & runtime readiness gating in FastAPI</span>
            </div>
          </div>
        </div>
      </div>

      {/* Pipeline Stepper */}
      <div className="p-6 rounded-xl border border-slate-200 bg-white shadow-2xs">
        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider mb-4">
          Training & Evaluation Lifecycle
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs">
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="font-mono text-indigo-600 font-bold block mb-1">STEP 01</span>
            <p className="font-semibold text-slate-800">Dataset Curation</p>
            <p className="text-[11px] text-slate-500 mt-1">Download and parse SROIE receipt annotations & bounding boxes.</p>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="font-mono text-indigo-600 font-bold block mb-1">STEP 02</span>
            <p className="font-semibold text-slate-800">Fine-Tuning</p>
            <p className="text-[11px] text-slate-500 mt-1">Train LayoutLMv3 token classifier using HuggingFace Transformers & PyTorch.</p>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="font-mono text-indigo-600 font-bold block mb-1">STEP 03</span>
            <p className="font-semibold text-slate-800">Benchmarking</p>
            <p className="text-[11px] text-slate-500 mt-1">Evaluate precision, recall, F1, and inference latency against Tesseract baseline.</p>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-200">
            <span className="font-mono text-indigo-600 font-bold block mb-1">STEP 04</span>
            <p className="font-semibold text-slate-800">Verified Serving</p>
            <p className="text-[11px] text-slate-500 mt-1">Validate checkpoint integrity & expose readiness status before enabling inference.</p>
          </div>
        </div>
      </div>
    </div>
  );
};
