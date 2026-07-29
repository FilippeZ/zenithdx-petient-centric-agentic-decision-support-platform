import DOMPurify from "dompurify";

const SanitizedHtml = ({ html }) => {
  const sanitizedHtml = DOMPurify.sanitize(html);
  return <div dangerouslySetInnerHTML={{ __html: sanitizedHtml }} />;
};

export default SanitizedHtml;
