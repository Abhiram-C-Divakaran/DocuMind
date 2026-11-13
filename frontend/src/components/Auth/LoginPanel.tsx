import React, { useState } from 'react';
import { Layers, Lock, Mail, AlertCircle, Loader2, ShieldCheck } from 'lucide-react';
import { apiClient } from '../../api/client';
import { User } from '../../types';

interface LoginPanelProps {
  onAuthenticated: (user: User) => void;
  sessionNotice?: string | null;
}

export const LoginPanel: React.FC<LoginPanelProps> = ({ onAuthenticated, sessionNotice }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitting(true);
    try {
      const tokenRes = await apiClient.login(email.trim(), password);
      onAuthenticated(tokenRes.user);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unable to sign in. Check credentials and try again.';
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6">
      <div className="w-full max-w-md bg-white rounded-2xl shadow-2xl border border-slate-200 overflow-hidden">
        <div className="px-8 py-6 bg-slate-900 text-white flex items-center gap-3.5">
          <div className="w-11 h-11 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <Layers className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight">DocuMind</h1>
            <p className="text-xs text-slate-400">Authenticated Document Intelligence</p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="p-8 space-y-5">
          <div>
            <h2 className="text-base font-bold text-slate-900">Sign in to Workspace</h2>
            <p className="text-xs text-slate-500 mt-1">
              Access document ingestion, asynchronous processing telemetry, and review queues.
            </p>
          </div>

          {sessionNotice && (
            <div className="p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-800 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-amber-600" />
              <span>{sessionNotice}</span>
            </div>
          )}

          {error && (
            <div className="p-3 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-700 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5 text-rose-600" />
              <span>{error}</span>
            </div>
          )}

          <div className="space-y-1.5">
            <label htmlFor="login-email" className="block text-xs font-semibold text-slate-700">
              Email Address
            </label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                id="login-email"
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="operator@example.com"
                className="w-full pl-9 pr-3 py-2.5 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label htmlFor="login-password" className="block text-xs font-semibold text-slate-700">
              Password
            </label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                id="login-password"
                type="password"
                required
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter account password"
                className="w-full pl-9 pr-3 py-2.5 text-sm border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 bg-indigo-600 hover:bg-indigo-700 text-white text-sm font-semibold rounded-lg shadow-sm transition-colors disabled:opacity-60"
          >
            {submitting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Verifying credentials...</span>
              </>
            ) : (
              <span>Sign In</span>
            )}
          </button>

          <div className="pt-3 border-t border-slate-100 flex items-center gap-2 text-[11px] text-slate-500">
            <ShieldCheck className="w-4 h-4 text-indigo-500 shrink-0" />
            <span>
              Provision accounts via your administrator or the CLI:
              <code className="ml-1 px-1 py-0.5 bg-slate-100 rounded text-slate-700">
                python -m backend.app.cli create-admin
              </code>
            </span>
          </div>
        </form>
      </div>
    </div>
  );
};

