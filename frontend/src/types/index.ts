export type DocumentStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export type ExtractionMethod = 'PLACEHOLDER' | 'TESSERACT_BASELINE' | 'LAYOUTLMV3';

export interface LineItem {
  description?: string | null;
  quantity?: number | string | null;
  unit_price?: number | string | null;
  total_price?: number | string | null;
  experimental?: boolean;
}

export interface StructuredExtractionData {
  vendor_name?: string | null;
  vendor_address?: string | null;
  document_date?: string | null;
  total_amount?: number | string | null;
  currency?: string | null;
  tax_amount?: number | string | null;
  line_items?: LineItem[];
}

export interface ExtractionResult {
  id: string;
  document_id: string;
  extraction_method: string;
  structured_data: StructuredExtractionData | null;
  confidence: number | null;
  field_confidences?: Record<string, number> | null;
  raw_text?: string | null;
  validation_errors: string[];
  extraction_metadata?: Record<string, unknown> | null;
  created_at: string;
}

export interface DocumentItem {
  id: string;
  original_filename: string;
  content_type: string;
  file_size_bytes: number;
  status: DocumentStatus;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
  extractions: ExtractionResult[];
}

export interface DocumentListResponse {
  items: DocumentItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface DocumentUploadResponse {
  document: DocumentItem;
  message: string;
}

export interface DashboardMetrics {
  total_documents: number;
  pending: number;
  processing: number;
  completed: number;
  failed: number;
}

export interface HealthStatus {
  status: string;
  service: string;
  environment: string;
  timestamp: string;
  components: {
    database: string;
    storage: string;
  };
  phase: number | string;
}

export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}
