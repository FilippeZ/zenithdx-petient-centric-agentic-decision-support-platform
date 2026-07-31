// frontend/src/components/LongitudinalGraphViewer.jsx
import React from 'react';

export default function LongitudinalGraphViewer({ patientId = '10000032', height = '650px' }) {
  const iframeSrc = `/api/v1/graph/patient/${patientId}/html`;

  return (
    <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 shadow-2xl">
      <iframe
        src={iframeSrc}
        width="100%"
        height={height}
        style={{ border: 'none' }}
        title={`Patient ${patientId} EHR Graph`}
        className="w-full rounded-2xl"
      />
    </div>
  );
}
