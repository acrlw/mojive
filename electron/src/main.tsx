import { createRoot } from 'react-dom/client';
import App from './App';
import './design-tokens.css';
import './styles.css';
import './scrollbars.css';
import { installScrollbarBehavior } from './scrollbars';
installScrollbarBehavior();
createRoot(document.getElementById('root')!).render(<App />);

import './workspace.css';
