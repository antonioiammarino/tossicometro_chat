import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';

interface ChatMessage {
  id: number;
  speaker: string;
  content: string;
}

interface ClassificationResult {
  explanation: string;
  is_toxic: boolean;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './chat.component.html',
  styleUrls: ['./chat.component.css']
})
export class ChatComponent {
  private readonly apiUrl = 'http://localhost:5001';

  constructor(private http: HttpClient) {}
  messages = signal<ChatMessage[]>([
    { id: 1, speaker: 'Persona1', content: '' },
    { id: 2, speaker: 'Persona2', content: '' }
  ]);

  dragActive = signal(false);
  uploadedImage = signal<string | null>(null);
  
  classificationResult = signal<ClassificationResult | null>(null);
  isClassifying = signal(false);
  classificationError = signal<string | null>(null);
  // Rating (1-5) for the classification result
  rating = signal<number>(0);

  setRating(value: number) {
    this.rating.set(value);
  }

  // Close the result modal and reset the classification result
  closeResultModal() {
    this.classificationResult.set(null);
    // reset rating and clean chat/image
    this.rating.set(0);
    this.cleanChat();
    this.removeImage();
  }

  // Close the error modal and reset the classification error
  closeErrorModal() {
    this.classificationError.set(null);
  }

  addMessage() {
    const currentMessages = this.messages();
    if (currentMessages.length < 6) {
      const lastMessage = currentMessages[currentMessages.length - 1];
      const nextSpeaker = lastMessage.speaker === 'Persona1' ? 'Persona2' : 'Persona1';
      
      const newMessage: ChatMessage = {
        id: currentMessages.length + 1,
        speaker: nextSpeaker,
        content: ''
      };
      this.messages.set([...currentMessages, newMessage]);
    }
  }

  removeMessage(id: number) {
    const currentMessages = this.messages();
    if (currentMessages.length > 2) {
      this.messages.set(currentMessages.filter(msg => msg.id !== id));
    }
  }

  updateMessage(id: number, content: string) {
    const currentMessages = this.messages();
    const updatedMessages = currentMessages.map(msg => 
      msg.id === id ? { ...msg, content } : msg
    );
    this.messages.set(updatedMessages);
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.dragActive.set(true);
  }

  onDragLeave(event: DragEvent) {
    event.preventDefault();
    this.dragActive.set(false);
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.dragActive.set(false);
    
    const files = event.dataTransfer?.files;
    if (files && files.length > 0) {
      this.handleImageUpload(files[0]);
    }
  }

  onFileSelected(event: Event) {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file) {
      this.handleImageUpload(file);
    }
  }

  private handleImageUpload(file: File) {
    if (file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        this.uploadedImage.set(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    }
  }

  removeImage() {
    this.uploadedImage.set(null);
  }

  cleanChat() {
    this.messages.set([
      { id: 1, speaker: 'Persona1', content: '' },
      { id: 2, speaker: 'Persona2', content: '' }
    ]);
  }

  hasChatContent(): boolean {
    return this.messages().some(msg => msg.content.trim() !== '');
  }

  canUploadImage(): boolean {
    return !this.hasChatContent();
  }

  canUseChat(): boolean {
    return !this.uploadedImage();
  }

  classifyConversation() {
    if (this.uploadedImage()) {
      this.classifyImage();
    } else {
      this.classifyChat();
    }
  }

  private classifyChat() {
    const messages = this.messages().filter(msg => msg.content.trim() !== '');
    
    // This is to prevent sending empty chats (Not permitted because the button should be disabled)
    if (messages.length === 0) {
      this.classificationError.set('Inserisci almeno un messaggio per la classificazione');
      return;
    }

    this.isClassifying.set(true);
    this.classificationError.set(null);
    this.classificationResult.set(null);

    this.http.post<ClassificationResult>(`${this.apiUrl}/classify`, { messages }).subscribe({
      next: (result) => {
        this.classificationResult.set(result);
        this.isClassifying.set(false);
      },
      error: (error) => {
        console.error('Errore classificazione:', error);
        this.classificationError.set('Errore durante la classificazione');
        this.isClassifying.set(false);
      }
    });
  }

  private classifyImage() {
    this.isClassifying.set(true);
    this.classificationError.set(null);
    this.classificationResult.set(null);

    this.http.post<ClassificationResult>(`${this.apiUrl}/classify-image`, { 
      image: this.uploadedImage() 
    }).subscribe({
      next: (result) => {
        this.classificationResult.set(result);
        this.isClassifying.set(false);
      },
      error: (error) => {
        console.error('Errore classificazione immagine:', error);
        this.classificationError.set(error.error?.error || 'Errore durante la classificazione dell\'immagine');
        this.isClassifying.set(false);
      }
    });
  }
}