import {
  DashboardMetrics,
  DocumentItem,
  DocumentListResponse,
  DocumentUploadResponse,
  HealthStatus,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

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
          errorMsg = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
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
    extractionMethod: string = 'PLACEHOLDER'
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (extractionMethod) {
      formData.append('extraction_method', extractionMethod);
    }

    const res = await fetch(`${API_BASE}/api/v1/documents`, {
      method: 'POST',
      body: formData,
    });

    return this.handleResponse<DocumentUploadResponse>(res);
  }
}

export const apiClient = new ApiClient();

