import React, { useState } from 'react';
import TypeWriter from './TypeWriter';
import { MdOutlineNavigateNext } from "react-icons/md";
import { IoSend } from 'react-icons/io5';
const questions = [
  {
    question: "Please enter your name:",
    type: "String",
    min: 2,
    max: 50
  },
  {
    question: "What is your current temperature (in Celsius)?",
    type: "Number",
    min: 35,
    max: 42
  },
  {
    question: "What is your heart rate (in bpm)?",
    type: "Number",
    min: 40,
    max: 180
  },
  {
    question: "What is your blood pressure (in mmHg)?",
    type: "String", // Blood pressure is typically given as a string, e.g., "120/80"
    min: 3,
    max: 7
  },
  {
    question: "What is your respiratory rate (in breaths per minute)?",
    type: "Number",
    min: 10,
    max: 40
  },
  {
    question: "What is your oxygen saturation (in %)?",
    type: "Number",
    min: 70,
    max: 100
  },
  {
    question: "What is your blood sugar level (in mg/dL)?",
    type: "Number",
    min: 70,
    max: 200
  }
];


const App = () => {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [formData, setFormData] = useState({
    name: '',
    temperature: '',
    heart_rate: '',
    blood_pressure: '',
    respiratory_rate: '',
    oxygen_saturation: '',
    blood_sugar: ''
  });
  const [submitted, setSubmitted] = useState(false);

  const [response,setResponse] = useState();
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value
    }));
  };

  const adjustHeight = (e) => {
    e.target.style.height = 'auto'; // Reset height
    e.target.style.height = `${Math.min(e.target.scrollHeight, 5 * 24)}px`; // 5 rows max (assuming 1 row = 24px)
  };
  const handleNext = () => {
    setCurrentQuestion(currentQuestion + 1);
  };

  const handleSubmit = async () => {
    try {
      const response = await fetch('http://localhost:8000/get_diagnosis/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: formData.name,
          temperature: parseFloat(formData.temperature),
          heart_rate: parseInt(formData.heart_rate, 10),
          blood_pressure: parseInt(formData.blood_pressure, 10),
          respiratory_rate: parseInt(formData.respiratory_rate, 10),
          oxygen_saturation: parseFloat(formData.oxygen_saturation),
          blood_sugar: parseFloat(formData.blood_sugar)
        })
      });
      const result = await response.json();
      setResponse(result);
      console.log('Diagnosis:', result);
      setSubmitted(true);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div>
      <h1 className='w-full bg-slate-300 p-10 text-3xl font-bold'>Health Diagnostic Form</h1>
      {!submitted ? (
        <div className='flex flex-col justify-center items-center h-[calc(100vh-8rem)] w-screen'>
          <TypeWriter data={questions[currentQuestion].question} />
          <div className="flex justify-center items-center w-[60%] py-4 px-8 bg-slate-200 rounded-xl">
          {questions[currentQuestion].type === "Number" ? (
  <input
    className="bg-slate-200 text-2xl outline-none w-full resize-none"
    type="number"
    name={Object.keys(formData)[currentQuestion]}
    value={formData[Object.keys(formData)[currentQuestion]]}
    min={questions[currentQuestion].min}
    max={questions[currentQuestion].max}
    required
    onChange={(e) => {
      handleChange(e);
      adjustHeight(e);
    }}
    onInput={adjustHeight}
  />
) : (
  <textarea
    rows={1}
    className="bg-slate-200 text-2xl outline-none w-full overflow-y-auto resize-none"
    name={Object.keys(formData)[currentQuestion]}
    value={formData[Object.keys(formData)[currentQuestion]]}
    required
    onChange={(e) => {
      handleChange(e);
      adjustHeight(e);
    }}
    onInput={adjustHeight}
  />
)}


      {currentQuestion < questions.length - 1 ? (
            <button className='pl-5' onClick={handleNext} disabled={!formData[Object.keys(formData)[currentQuestion]]}>
              <MdOutlineNavigateNext size={40} />
            </button>
          ) : (
            <button onClick={handleSubmit}><IoSend size={30} /></button>
          )}
    </div>
          
        </div>
      ) : (
        <div className='flex flex-col h-[calc(100vh-8rem)] w-screen justify-center items-center'>
          <h2 className='text-5xl'>Diagnosis: "{response?.diagnosis}"</h2>
          <p className='m-4 text-3xl w-[80%] text-center'>Advice: {response?.advice}</p>
        </div>
      )}
    </div>
  );
};

export default App;
