import React, { useEffect, useState } from 'react';

function TypeWriter({ data }) {
  const [currentWord, setCurrentWord] = useState("");

  useEffect(() => {
    if (!data) {
      setCurrentWord(""); // Reset if the data is empty
      return;
    }

    let j = 0;
    let timeoutId;

    const type = () => {
      if (j < data.length) {
        setCurrentWord(data.substring(0, j + 1));
        j++;
        timeoutId = setTimeout(type, 100); // Typing speed
      }
    };

    type();
    return () => clearTimeout(timeoutId);
  }, [data]);

  const renderWithNewlinesAndTabs = (text) => {
    return text.split('\n').map((line, index) => (
      <React.Fragment key={index}>
        {line.split('\t').map((part, tabIndex) => (
          <React.Fragment key={tabIndex}>
            {part}
            {tabIndex < line.split('\t').length - 1 && <span style={{ display: 'inline-block', width: '2ch' }}></span>} {/* Add space for tab */}
          </React.Fragment>
        ))}
        <br />
      </React.Fragment>
    ));
  };

  return (
    <div>
      <h1 className='w-full text-md font-bold inline-block font-SpaceMono lg:leading-loose'>
        {renderWithNewlinesAndTabs(currentWord)}
        <span className='inline-block ml-2 animate-blink md:h-8 h-5 w-[2px] '></span>
      </h1>
    </div>
  );
}

export default TypeWriter;
