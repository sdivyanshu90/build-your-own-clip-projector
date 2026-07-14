import http from 'k6/http';
import { check } from 'k6';
export const options = { stages: [{ duration: '1m', target: 25 }, { duration: '3m', target: 100 }, { duration: '1m', target: 0 }], thresholds: { http_req_failed: ['rate<0.01'], http_req_duration: ['p(95)<500'] } };
export default function () { const response = http.post(`${__ENV.BASE_URL}/v1/embeddings/text`, JSON.stringify({ texts: ['a photo of a cat'] }), { headers: { 'Content-Type': 'application/json', 'X-API-Key': __ENV.API_KEY } }); check(response, { '200': r => r.status === 200 }); }
