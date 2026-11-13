export type DocumentStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export type ReviewStatus =
  | 'NOT_REQUIRED'
  | 'PENDING_REVIEW'
  | 'IN_REVIEW'
  | 'APPROVED'
  | 'REJECTED';

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
  review_status: ReviewStatus;
  needs_review: boolean;
  review_reasons: string[];
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

export interface ReviewLineItemPayload {
  description?: string | null;
  quantity?: string | number | null;
  unit_price?: string | number | null;
  total_price?: string | number | null;
}

export interface ReviewCorrectionPayload {
  vendor_name?: string | null;
  vendor_address?: string | null;
  document_date?: string | null;
  total_amount?: string | number | null;
  currency?: string | null;
  tax_amount?: string | number | null;
  line_items?: ReviewLineItemPayload[];
}

export interface DocumentReview {
  id?: string | null;
  document_id: string;
  extraction_result_id?: string | null;
  review_status: ReviewStatus;
  document_status: DocumentStatus;
  extraction_method?: string | null;
  confidence?: number | null;
  field_confidences?: Record<string, number> | null;
  validation_errors: string[];
  review_reasons: string[];
  missing_critical_fields: string[];
  original_data?: StructuredExtractionData | null;
  corrected_data?: StructuredExtractionData | null;
  changed_fields: string[];
  raw_text?: string | null;
  reviewer_identifier: string;
  reviewer_notes?: string | null;
  version: number;
  created_at?: string | null;
  updated_at?: string | null;
  approved_at?: string | null;
}

export interface ReviewQueueItem {
  document_id: string;
  review_id?: string | null;
  original_filename: string;
  content_type: string;
  file_size_bytes: number;
  document_status: DocumentStatus;
  review_status: ReviewStatus;
  needs_review: boolean;
  review_reasons: string[];
  missing_critical_fields: string[];
  extraction_method?: string | null;
  confidence?: number | null;
  validation_errors_count: number;
  changed_fields: string[];
  reviewer_identifier: string;
  version: number;
  created_at: string;
  updated_at: string;
  approved_at?: string | null;
}

export interface ReviewListResponse {
  items: ReviewQueueItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface FieldCorrectionItem {
  id: string;
  review_id: string;
  field_name: string;
  original_value?: string | null;
  corrected_value?: string | null;
  created_at: string;
}

export interface ReviewAuditEventItem {
  id: string;
  review_id: string;
  action: string;
  from_status?: string | null;
  to_status?: string | null;
  changed_fields: string[];
  actor_identifier: string;
  notes?: string | null;
  event_timestamp: string;
}

export interface ReviewHistoryResponse {
  document_id: string;
  review_id?: string | null;
  review_status: ReviewStatus;
  field_corrections: FieldCorrectionItem[];
  audit_events: ReviewAuditEventItem[];
}

export interface ReviewMetrics {
  total_documents: number;
  not_required: number;
  pending_review: number;
  in_review: number;
  awaiting_review: number;
  approved: number;
  rejected: number;
  avg_corrected_fields: number;
  review_completion_rate: number;
  eligible_for_feedback_export: number;
  review_confidence_threshold: number;
}

export interface PreviewMeta {
  document_id: string;
  original_filename: string;
  content_type: string;
  is_pdf: boolean;
  total_pages: number;
  max_preview_pages: number;
  preview_url: string;
}
