import {
  DashboardMetrics,
  DocumentItem,
  DocumentListResponse,
  DocumentReview,
  DocumentUploadResponse,
  HealthStatus,
  PreviewMeta,
  ReviewCorrectionPayload,
  ReviewHistoryResponse,
  ReviewListResponse,
  ReviewMetrics,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

export interface ReviewDecisionPayload {
  corrected_data?: ReviewCorrectionPayload;
  reviewer_notes?: string | null;
  reviewer_identifier?: string | null;
  expected_version?: number | null;
}

class ApiClient {
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

  async checkHealth(): Promise<HealthStatus> {
    const res = await fetch(`${API_BASE}/health`);
    return this.handleResponse<HealthStatus>(res);
  }

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
    const res = await fetch(`${API_BASE}/api/v1/documents${queryString}`);
    return this.handleResponse<DocumentListResponse>(res);
  }

  async fetchDocument(id: string): Promise<DocumentItem> {
    const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(id)}`);
    return this.handleResponse<DocumentItem>(res);
  }

  async fetchMetrics(): Promise<DashboardMetrics> {
    const res = await fetch(`${API_BASE}/api/v1/documents/metrics/summary`);
    return this.handleResponse<DashboardMetrics>(res);
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

    const res = await fetch(`${API_BASE}/api/v1/documents`, {
      method: 'POST',
      body: formData,
    });

    return this.handleResponse<DocumentUploadResponse>(res);
  }

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
    const res = await fetch(`${API_BASE}/api/v1/reviews${queryString}`);
    return this.handleResponse<ReviewListResponse>(res);
  }

  async fetchReviewMetrics(): Promise<ReviewMetrics> {
    const res = await fetch(`${API_BASE}/api/v1/reviews/metrics`);
    return this.handleResponse<ReviewMetrics>(res);
  }

  async fetchDocumentReview(documentId: string): Promise<DocumentReview> {
    const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review`);
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
    const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return this.handleResponse<DocumentReview>(res);
  }

  async approveReview(documentId: string, payload?: ReviewDecisionPayload): Promise<DocumentReview> {
    const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload ?? {}),
    });
    return this.handleResponse<DocumentReview>(res);
  }

  async rejectReview(documentId: string, payload?: ReviewDecisionPayload): Promise<DocumentReview> {
    const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload ?? {}),
    });
    return this.handleResponse<DocumentReview>(res);
  }

  async reopenReview(documentId: string, payload?: ReviewDecisionPayload): Promise<DocumentReview> {
    const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review/reopen`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload ?? {}),
    });
    return this.handleResponse<DocumentReview>(res);
  }

  async fetchReviewHistory(documentId: string): Promise<ReviewHistoryResponse> {
    const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/review/history`);
    return this.handleResponse<ReviewHistoryResponse>(res);
  }

  async fetchPreviewMeta(documentId: string): Promise<PreviewMeta> {
    const res = await fetch(`${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/preview/meta`);
    return this.handleResponse<PreviewMeta>(res);
  }

  getDocumentPreviewUrl(documentId: string, page?: number): string {
    const base = `${API_BASE}/api/v1/documents/${encodeURIComponent(documentId)}/file`;
    return page !== undefined ? `${base}?page=${page}` : base;
  }
}

export const apiClient = new ApiClient();

