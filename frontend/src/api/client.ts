import {
  DashboardMetrics,
  DocumentItem,
  DocumentListResponse,
  DocumentReview,
  DocumentStatusResponse,
  DocumentUploadResponse,
  HealthStatus,
  ModelReadinessResponse,
  PreviewMeta,
  ProcessingJob,
  ProcessingJobMetrics,
  ReviewCorrectionPayload,
  ReviewHistoryResponse,
  ReviewListResponse,
  ReviewMetrics,
  TokenResponse,
  User,
  UserRole,
} from '../types';
import { authStorage } from './authStorage';

const API_BASE = import.meta.env.VITE_API_URL || '';

export interface ReviewDecisionPayload {
  corrected_data?: ReviewCorrectionPayload;
  reviewer_notes?: string | null;
  reviewer_identifier?: string | null;
  expected_version?: number | null;
}

class ApiClient {
  private refreshPromise: Promise<boolean> | null = null;
  private onSessionExpired: (() => void) | null = null;

  setSessionExpiredHandler(handler: (() => void) | null): void {
    this.onSessionExpired = handler;
  }

  private async tryRefreshSession(): Promise<boolean> {
    if (this.refreshPromise) {
      return this.refreshPromise;
    }
    const refreshToken = authStorage.getRefreshToken();
    if (!refreshToken) {
      return false;
    }

    this.refreshPromise = (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/v1/auth/refresh`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ refresh_token: refreshToken }),
        });
        if (!res.ok) {
          authStorage.clear();
          return false;
        }
        const tokens: TokenResponse = await res.json();
        authStorage.setTokens(tokens.access_token, tokens.refresh_token);
        return true;
      } catch {
        return false;
      } finally {
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  private async authFetch(input: string, init: RequestInit = {}, retryOn401 = true): Promise<Response> {
    const headers = new Headers(init.headers || {});
    const token = authStorage.getAccessToken();
    if (token) {
      headers.set('Authorization', `Bearer ${token}`);
    }

    const res = await fetch(input, { ...init, headers });
    if (res.status === 401 && retryOn401) {
      const refreshed = await this.tryRefreshSession();
      if (refreshed) {
        return this.authFetch(input, init, false);
      }
      authStorage.clear();
      if (this.onSessionExpired) {
        this.onSessionExpired();
      }
    }
    return res;
  }

  private async handleResponse<T>(res: Response): Promise<T> {
    if (!res.ok) {
      let errorMsg = `Server error (${res.status})`;
      let errorDetails: unknown = undefined;

      try {
        const errorJson = await res.json();
        if (errorJson.error) {
          errorMsg = errorJson.error.message || errorMsg;
          errorDetails = errorJson.error.details;
        } else if (errorJson.detail) {
          if (typeof errorJson.detail === 'string') {
            errorMsg = errorJson.detail;
          } else if (Array.isArray(errorJson.detail)) {
            errorMsg = errorJson.detail
              .map((d: { loc?: Array<string | number>; msg?: string }) => {
                const path = d.loc?.filter((p) => p !== 'body').join('.') || 'field';
                return `${path}: ${d.msg || 'Invalid value'}`;
              })
              .join(' | ');
            errorDetails = errorJson.detail;
          } else {
            errorMsg = JSON.stringify(errorJson.detail);
          }
        }
      } catch {
        // Response wasn't JSON
      }

      const err = new Error(errorMsg) as Error & { status: number; details?: unknown };
      err.status = res.status;
      err.details = errorDetails;
      throw err;
    }
    return res.json();
  }

  // --- Auth & Admin Users ---
  async login(email: string, password: string): Promise<TokenResponse> {
    const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const tokens = await this.handleResponse<TokenResponse>(res);
    authStorage.setTokens(tokens.access_token, tokens.refresh_token);
    return tokens;
  }

  async logout(): Promise<void> {
    const refreshToken = authStorage.getRefreshToken();
    try {
      if (authStorage.getAccessToken()) {
        await this.authFetch(
          `${API_BASE}/api/v1/auth/logout`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ refresh_token: refreshToken }),
          },
          false
        );
      }
    } finally {
      authStorage.clear();
    }
  }

  async getMe(): Promise<User> {
    const res = await this.authFetch(`${API_BASE}/api/v1/auth/me`);
    return this.handleResponse<User>(res);
  }

  async listUsers(): Promise<{ items: User[]; total: number }> {
    const res = await this.authFetch(`${API_BASE}/api/v1/users`);
    return this.handleResponse<{ items: User[]; total: number }>(res);
  }

  async createUser(payload: {
    email: string;
    password: string;
    full_name?: string;
    role: UserRole;
  }): Promise<User> {
    const res = await this.authFetch(`${API_BASE}/api/v1/users`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return this.handleResponse<User>(res);
  }

  async updateUser(
    userId: string,
    payload: { full_name?: string; role?: UserRole; is_active?: boolean; password?: string }
  ): Promise<User> {
    const res = await this.authFetch(`${API_BASE}/api/v1/users/${encodeURIComponent(userId)}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return this.handleResponse<User>(res);
  }

  // --- Health & Readiness ---
  async checkHealth(): Promise<HealthStatus> {
    const res = await fetch(`${API_BASE}/health`);
    return this.handleResponse<HealthStatus>(res);
  }

  async fetchModelReadiness(): Promise<ModelReadinessResponse> {
    const res = await fetch(`${API_BASE}/api/v1/models/readiness`);
    return this.handleResponse<ModelReadinessResponse>(res);
  }

  // --- Documents & Async Jobs ---
  async fetchDocuments(params?: {
    page?: number;
    page_size?: number;
    status?: string;
    search?: string;
  }): Promise<DocumentListResponse> {
    const query = new URLSearchParams();
    if (params?.page) query.append('page', params.page.toString());
    if (params?.page_size) query.append('page_size', params.page_size.toString());
    if (params?.status) query.append('status', params.status);
    if (params?.search) query.append('search', params.search);

    const queryString = query.toString() ? `?${query.toString()}` : '';
    const res = await this.authFetch(`${API_BASE}/api/v1/documents${queryString}`);
    return this.handleResponse<DocumentListResponse>(res);
  }

  async fetchDocument(id: string): Promise<DocumentItem> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(id)}`);
    return this.handleResponse<DocumentItem>(res);
  }

  async fetchDocumentStatus(id: string): Promise<DocumentStatusResponse> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(id)}/status`);
    return this.handleResponse<DocumentStatusResponse>(res);
  }

  async retryDocument(id: string, extractionMethod?: string): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    if (extractionMethod) {
      formData.append('extraction_method', extractionMethod);
    }
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(id)}/retry`, {
      method: 'POST',
      body: extractionMethod ? formData : undefined,
    });
    return this.handleResponse<DocumentUploadResponse>(res);
  }

  async cancelDocument(id: string): Promise<ProcessingJob> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(id)}/cancel`, {
      method: 'POST',
    });
    return this.handleResponse<ProcessingJob>(res);
  }

  async deleteDocument(id: string): Promise<{ status: string; document_id: string }> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    return this.handleResponse<{ status: string; document_id: string }>(res);
  }

  async fetchMetrics(): Promise<DashboardMetrics> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/metrics/summary`);
    return this.handleResponse<DashboardMetrics>(res);
  }

  async getJobMetrics(): Promise<ProcessingJobMetrics> {
    const res = await this.authFetch(`${API_BASE}/api/v1/jobs/metrics`);
    return this.handleResponse<ProcessingJobMetrics>(res);
  }

  async triggerJobRecovery(): Promise<{
    republished_outbox: number;
    requeued_stale_jobs: number;
    failed_stale_jobs: number;
  }> {
    const res = await this.authFetch(`${API_BASE}/api/v1/jobs/recover`, { method: 'POST' });
    return this.handleResponse(res);
  }

  async exportFeedbackAdmin(): Promise<{ exported_count: number }> {
    const res = await this.authFetch(`${API_BASE}/api/v1/reviews/feedback-export`, { method: 'POST' });
    return this.handleResponse<{ exported_count: number }>(res);
  }

  async uploadDocument(
    file: File,
    extractionMethod: string = 'TESSERACT_BASELINE',
    requestReview: boolean = false
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (extractionMethod) {
      formData.append('extraction_method', extractionMethod);
    }
    if (requestReview) {
      formData.append('request_review', 'true');
    }

    const res = await this.authFetch(`${API_BASE}/api/v1/documents`, {
      method: 'POST',
      body: formData,
    });

    return this.handleResponse<DocumentUploadResponse>(res);
  }

  // --- Human-in-the-Loop Reviews ---
  async fetchReviewQueue(params?: {
    page?: number;
    page_size?: number;
    review_status?: string;
    extraction_method?: string;
    confidence_max?: number;
    missing_field?: string;
    search?: string;
    sort_by?: string;
  }): Promise<ReviewListResponse> {
    const query = new URLSearchParams();
    if (params?.page) query.append('page', params.page.toString());
    if (params?.page_size) query.append('page_size', params.page_size.toString());
    if (params?.review_status) query.append('review_status', params.review_status);
    if (params?.extraction_method) query.append('extraction_method', params.extraction_method);
    if (params?.confidence_max !== undefined) query.append('confidence_max', params.confidence_max.toString());
    if (params?.missing_field) query.append('missing_field', params.missing_field);
    if (params?.search) query.append('search', params.search);
    if (params?.sort_by) query.append('sort_by', params.sort_by);

    const queryString = query.toString() ? `?${query.toString()}` : '';
    const res = await this.authFetch(`${API_BASE}/api/v1/reviews${queryString}`);
    return this.handleResponse<ReviewListResponse>(res);
  }

  async fetchReviewMetrics(): Promise<ReviewMetrics> {
    const res = await this.authFetch(`${API_BASE}/api/v1/reviews/metrics`);
    return this.handleResponse<ReviewMetrics>(res);
  }

  async fetchDocumentReview(documentId: string): Promise<DocumentReview> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review`);
    return this.handleResponse<DocumentReview>(res);
  }

  async saveReviewDraft(
    documentId: string,
    payload: {
      corrected_data: ReviewCorrectionPayload;
      reviewer_notes?: string | null;
      reviewer_identifier?: string | null;
      expected_version?: number | null;
    }
  ): Promise<DocumentReview> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return this.handleResponse<DocumentReview>(res);
  }

  async approveReview(documentId: string, payload?: ReviewDecisionPayload): Promise<DocumentReview> {
    const res = await this.authFetch(
      `${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review/approve`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload ?? {}),
      }
    );
    return this.handleResponse<DocumentReview>(res);
  }

  async rejectReview(documentId: string, payload?: ReviewDecisionPayload): Promise<DocumentReview> {
    const res = await this.authFetch(
      `${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review/reject`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload ?? {}),
      }
    );
    return this.handleResponse<DocumentReview>(res);
  }

  async reopenReview(documentId: string, payload?: ReviewDecisionPayload): Promise<DocumentReview> {
    const res = await this.authFetch(
      `${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review/reopen`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload ?? {}),
      }
    );
    return this.handleResponse<DocumentReview>(res);
  }

  async fetchReviewHistory(documentId: string): Promise<ReviewHistoryResponse> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review/history`);
    return this.handleResponse<ReviewHistoryResponse>(res);
  }

  async fetchPreviewMeta(documentId: string): Promise<PreviewMeta> {
    const res = await this.authFetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/preview/meta`);
    return this.handleResponse<PreviewMeta>(res);
  }

  getDocumentPreviewUrl(documentId: string, page?: number): string {
    const base = `${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/file`;
    return page !== undefined ? `${base}?page=${page}` : base;
  }

  async fetchDocumentPreviewBlobUrl(documentId: string, page?: number): Promise<string> {
    const res = await this.authFetch(this.getDocumentPreviewUrl(documentId, page));
    if (!res.ok) {
      await this.handleResponse(res);
    }
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  }
}

export const apiClient = new ApiClient();
