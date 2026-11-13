import React, { useEffect, useState } from 'react';
import {
  UserPlus,
  Users,
  Shield,
  RefreshCw,
  Download,
  CheckCircle2,
  XCircle,
  Loader2,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import { HealthStatus, ProcessingJobMetrics, User, UserRole } from '../../types';

interface UserManagementViewProps {
  currentUser: User;
  health?: HealthStatus | null;
  onRefreshHealth?: () => Promise<void> | void;
  onToast: (type: 'success' | 'error' | 'info', msg: string) => void;
}

export const UserManagementView: React.FC<UserManagementViewProps> = ({
  currentUser,
  health = null,
  onRefreshHealth,
  onToast,
}) => {
  const [users, setUsers] = useState<User[]>([]);
  const [jobMetrics, setJobMetrics] = useState<ProcessingJobMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);

  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<UserRole>('REVIEWER');

  const loadData = async () => {
    setLoading(true);
    try {
      const [usersRes, metricsRes] = await Promise.all([
        apiClient.listUsers(),
        apiClient.getJobMetrics(),
      ]);
      setUsers(usersRes.items);
      setJobMetrics(metricsRes);
      if (onRefreshHealth) {
        await onRefreshHealth();
      }
    } catch (err: unknown) {
      onToast('error', err instanceof Error ? err.message : 'Failed to load admin console data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    try {
      await apiClient.createUser({
        email: email.trim(),
        password,
        full_name: fullName.trim() || undefined,
        role,
      });
      onToast('success', `Account '${email.trim()}' created with role ${role}.`);
      setEmail('');
      setFullName('');
      setPassword('');
      await loadData();
    } catch (err: unknown) {
      onToast('error', err instanceof Error ? err.message : 'Failed to create user.');
    } finally {
      setCreating(false);
    }
  };

  const handleToggleActive = async (target: User) => {
    try {
      await apiClient.updateUser(target.id, { is_active: !target.is_active });
      onToast('success', `${target.email} is now ${!target.is_active ? 'active' : 'disabled'}.`);
      await loadData();
    } catch (err: unknown) {
      onToast('error', err instanceof Error ? err.message : 'Failed to update account.');
    }
  };

  const handleRecoverJobs = async () => {
    try {
      const res = await apiClient.triggerJobRecovery();
      onToast(
        'info',
        `Recovery run: republished=${res.republished_outbox}, requeued=${res.requeued_stale_jobs}, failed_stale=${res.failed_stale_jobs}`
      );
      await loadData();
    } catch (err: unknown) {
      onToast('error', err instanceof Error ? err.message : 'Job recovery failed.');
    }
  };

  const handleExportFeedback = async () => {
    try {
      const res = await apiClient.exportFeedbackAdmin();
      onToast('success', `Exported ${res.exported_count} approved feedback records to active-learning JSONL.`);
    } catch (err: unknown) {
      onToast('error', err instanceof Error ? err.message : 'Feedback export failed.');
    }
  };

  return (
    <div className="space-y-6">
      {/* Admin Operations & Job Telemetry */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Shield className="w-4 h-4 text-indigo-600" />
              Queue Operations & Active Learning Export
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              PostgreSQL-backed job metrics, outbox/stale job recovery, and admin-only feedback export
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={loadData}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 rounded-lg border border-slate-200 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Refresh Telemetry
            </button>
            <button
              onClick={handleRecoverJobs}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg border border-slate-200 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Run Outbox & Stale Recovery
            </button>
            <button
              onClick={handleExportFeedback}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Export Approved Feedback
            </button>
          </div>
        </div>

        {/* Live System Readiness Strip (real DB / Redis / Worker / OCR / ML checks) */}
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2.5 text-xs">
          <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50/70 flex items-center justify-between">
            <span className="text-slate-500">Database</span>
            <span className={`font-semibold ${health?.database === 'connected' ? 'text-emerald-700' : 'text-rose-700'}`}>
              {health?.database ?? 'unknown'}
            </span>
          </div>
          <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50/70 flex items-center justify-between">
            <span className="text-slate-500">Redis Broker</span>
            <span className={`font-semibold ${health?.redis === 'connected' ? 'text-emerald-700' : 'text-amber-700'}`}>
              {health?.redis ?? 'unknown'}
            </span>
          </div>
          <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50/70 flex items-center justify-between">
            <span className="text-slate-500">Celery Worker</span>
            <span
              className={`font-semibold ${
                health?.worker === 'available'
                  ? 'text-emerald-700'
                  : health?.worker === 'unknown'
                  ? 'text-amber-700'
                  : 'text-rose-700'
              }`}
            >
              {health?.worker ?? 'unknown'}
            </span>
          </div>
          <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50/70 flex items-center justify-between">
            <span className="text-slate-500">Tesseract OCR</span>
            <span className={`font-semibold ${health?.tesseract_available ? 'text-emerald-700' : 'text-amber-700'}`}>
              {health ? (health.tesseract_available ? 'ready' : 'unavailable') : 'unknown'}
            </span>
          </div>
          <div className="p-2.5 rounded-lg border border-slate-200 bg-slate-50/70 flex items-center justify-between">
            <span className="text-slate-500">LayoutLMv3</span>
            <span
              className={`font-semibold ${
                health?.layoutlmv3_is_production_ready
                  ? 'text-emerald-700'
                  : health?.layoutlmv3_ready
                  ? 'text-amber-700'
                  : 'text-slate-600'
              }`}
            >
              {health
                ? health.layoutlmv3_is_production_ready
                  ? 'ready'
                  : health.layoutlmv3_ready
                  ? 'smoke-only'
                  : 'unavailable'
                : 'unknown'}
            </span>
          </div>
        </div>

        {jobMetrics && (
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3 text-xs">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <div className="text-slate-500">Total Jobs</div>
              <div className="text-lg font-bold text-slate-900">{jobMetrics.total_jobs}</div>
            </div>
            <div className="p-3 bg-amber-50 rounded-lg border border-amber-200">
              <div className="text-amber-700">Queued</div>
              <div className="text-lg font-bold text-amber-900">{jobMetrics.queued}</div>
            </div>
            <div className="p-3 bg-orange-50 rounded-lg border border-orange-200">
              <div className="text-orange-700">Retrying</div>
              <div className="text-lg font-bold text-orange-900">{jobMetrics.retrying}</div>
            </div>
            <div className="p-3 bg-sky-50 rounded-lg border border-sky-200">
              <div className="text-sky-700">Processing</div>
              <div className="text-lg font-bold text-sky-900">{jobMetrics.processing}</div>
            </div>
            <div className="p-3 bg-emerald-50 rounded-lg border border-emerald-200">
              <div className="text-emerald-700">Completed</div>
              <div className="text-lg font-bold text-emerald-900">{jobMetrics.completed}</div>
            </div>
            <div className="p-3 bg-rose-50 rounded-lg border border-rose-200">
              <div className="text-rose-700">Failed / Canc.</div>
              <div className="text-lg font-bold text-rose-900">
                {jobMetrics.failed + jobMetrics.cancelled}
              </div>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <div className="text-slate-500">Avg Duration</div>
              <div className="text-lg font-bold text-slate-900">
                {jobMetrics.average_duration_seconds != null
                  ? `${jobMetrics.average_duration_seconds.toFixed(2)}s`
                  : '—'}
              </div>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
              <div className="text-slate-500">Outbox / Stale</div>
              <div className="text-lg font-bold text-slate-900">
                {jobMetrics.unpublished_outbox_events} / {jobMetrics.stale_running_jobs}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Create User Form */}
      <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs">
        <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2 mb-4">
          <UserPlus className="w-4 h-4 text-indigo-600" />
          Provision User Account
        </h3>
        <form onSubmit={handleCreateUser} className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <input
            type="email"
            required
            placeholder="Email address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="px-3 py-2 text-xs border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <input
            type="text"
            placeholder="Full name (optional)"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            className="px-3 py-2 text-xs border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <input
            type="password"
            required
            minLength={8}
            placeholder="Password (>=8 chars)"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="px-3 py-2 text-xs border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500"
          />
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as UserRole)}
            className="px-3 py-2 text-xs border border-slate-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="REVIEWER">REVIEWER (Upload & Review)</option>
            <option value="VIEWER">VIEWER (Read-Only)</option>
            <option value="ADMIN">ADMIN (Full Access)</option>
          </select>
          <button
            type="submit"
            disabled={creating}
            className="flex items-center justify-center gap-1.5 px-4 py-2 text-xs font-semibold text-white bg-indigo-600 hover:bg-indigo-700 rounded-lg disabled:opacity-50"
          >
            {creating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <UserPlus className="w-3.5 h-3.5" />}
            Create Account
          </button>
        </form>
      </div>

      {/* Users Table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-2xs">
        <div className="px-6 py-4 border-b border-slate-200 flex items-center justify-between">
          <h3 className="text-sm font-bold text-slate-900 flex items-center gap-2">
            <Users className="w-4 h-4 text-slate-500" />
            Registered Accounts ({users.length})
          </h3>
        </div>
        {loading ? (
          <div className="p-8 text-center text-xs text-slate-500">Loading accounts...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 uppercase text-[11px]">
                <tr>
                  <th className="px-6 py-3">User</th>
                  <th className="px-6 py-3">Role</th>
                  <th className="px-6 py-3">Status</th>
                  <th className="px-6 py-3">Last Login</th>
                  <th className="px-6 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {users.map((u) => (
                  <tr key={u.id} className="hover:bg-slate-50">
                    <td className="px-6 py-3.5">
                      <div className="font-semibold text-slate-900">{u.email}</div>
                      {u.full_name && <div className="text-[11px] text-slate-500">{u.full_name}</div>}
                    </td>
                    <td className="px-6 py-3.5">
                      <span className="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                        {u.role}
                      </span>
                    </td>
                    <td className="px-6 py-3.5">
                      {u.is_active ? (
                        <span className="inline-flex items-center gap-1 text-emerald-700 font-medium">
                          <CheckCircle2 className="w-3.5 h-3.5" /> Active
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-rose-700 font-medium">
                          <XCircle className="w-3.5 h-3.5" /> Disabled
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-3.5 text-slate-500">
                      {u.last_login_at ? new Date(u.last_login_at).toLocaleString() : 'Never'}
                    </td>
                    <td className="px-6 py-3.5 text-right">
                      {u.id !== currentUser.id ? (
                        <button
                          onClick={() => handleToggleActive(u)}
                          className="px-2.5 py-1 text-[11px] font-semibold rounded border border-slate-200 hover:bg-slate-100 text-slate-700"
                        >
                          {u.is_active ? 'Disable' : 'Enable'}
                        </button>
                      ) : (
                        <span className="text-[11px] text-slate-400 italic">Current user</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

